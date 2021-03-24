#pragma once

#include "ghex_libfabric_defines.hpp"
#include <ghex/transport_layer/libfabric/libfabric_macros.hpp>
#include <ghex/transport_layer/libfabric/rma/atomic_count.hpp>
//
#include <ghex/transport_layer/libfabric/rma/detail/memory_region_impl.hpp>
#include <ghex/transport_layer/libfabric/performance_counter.hpp>
//
#include <boost/lockfree/stack.hpp>
//
#include <atomic>
#include <stack>
#include <unordered_map>
#include <iostream>
#include <cstddef>
#include <memory>
#include <mutex>
#include <array>
#include <sstream>
#include <string>

// Define this to track which regions were not returned to the pool after use
// #define RMA_POOL_DEBUG_SET 1

#if RMA_POOL_DEBUG_SET
# include <mutex>
# include <set>
#endif

namespace gridtools { namespace ghex {
    // cppcheck-suppress ConfigurationNotChecked
static hpx::debug::enable_print<false> mps_deb("MPSTACK");
static hpx::debug::enable_print<true>  mps_err("MPSTACK");
}}

namespace gridtools {
namespace ghex {
namespace tl {
namespace libfabric {
namespace rma {
namespace detail
{

    namespace bl = boost::lockfree;

    // A simple tag type we use for logging assistance (identification)
    struct pool_tiny   { static const char *desc() { return "Tiny ";   } };
    struct pool_small  { static const char *desc() { return "Small ";  } };
    struct pool_medium { static const char *desc() { return "Medium "; } };
    struct pool_large  { static const char *desc() { return "Large ";  } };

    // ---------------------------------------------------------------------------
    // memory pool stack is responsible for allocating large blocks of memory
    // from the system heap and splitting them into N small equally sized region/blocks
    // that are then stored in a stack and handed out on demand when a block of an
    // appropriate size is required.
    //
    // The memory pool class maintains N of these stacks for different sized blocks
    // ---------------------------------------------------------------------------
    template <typename RegionProvider,
              typename Allocator,
              typename PoolType,
              std::size_t ChunkSize>
    struct memory_pool_stack
    {
        typedef typename RegionProvider::provider_domain domain_type;
        typedef memory_region_impl<RegionProvider>       region_type;
        typedef std::shared_ptr<region_type>             region_ptr;

        // ------------------------------------------------------------------------
        memory_pool_stack(domain_type *pd, int num_initial_chunks) :
            accesses_(0), in_use_(0), chunks_avail_(0), pd_(pd), free_list_(num_initial_chunks)
        {
            allocate_pool(num_initial_chunks);
        }

        // ------------------------------------------------------------------------
        bool allocate_pool(uint32_t num_chunks)
        {
            GHEX_DP_ONLY(mps_deb, trace(hpx::debug::str<>(PoolType::desc()), "Allocating"
               , "ChunkSize", hpx::debug::hex<4>(ChunkSize)
               , "num_chunks", hpx::debug::dec<>(num_chunks)));

            // Allocate one very large registered block for N small blocks
            region_ptr block =
                Allocator().malloc(pd_, ChunkSize*num_chunks);

            // store a copy of this to make sure it is 'alive'
            block_list_[block->get_address()] = block;

            // add this many chunks to the tracking totals
            in_use_ += num_chunks;
            chunks_avail_ += num_chunks;
            region_list_.reserve(chunks_avail_);

            // break the large region into N small regions
            uint64_t offset = 0;
            for (std::size_t i=0; i<num_chunks; ++i) {
                // we must keep a copy of the sub-region since we only pass
                // pointers to regions around the code.
                region_type *new_region = new region_type(
                    block->get_region(),
                    static_cast<char*>(block->get_base_address()) + offset,
                    static_cast<char*>(block->get_base_address()),
                    ChunkSize,
                    region_type::BLOCK_PARTIAL
                );
                region_list_.push_back(new_region);
                GHEX_DP_ONLY(mps_deb, trace(hpx::debug::str<>(PoolType::desc()), "Allocate Block"
                   , hpx::debug::dec<>(i)
                   , new_region));
                // push the pointer onto our stack
                push(new_region);
                offset += ChunkSize;
            }
            return true;
        }

        // ------------------------------------------------------------------------
        void DeallocatePool()
        {
            if (in_use_!=0) {
                GHEX_DP_ONLY(mps_err, trace(hpx::debug::str<>(PoolType::desc())
                   , "Deallocating free_list : Not all blocks were returned"
                   , "refcounts", hpx::debug::dec<>(in_use_)));
            }
#ifdef RMA_POOL_DEBUG_SET
            for (auto region : region_set_) {
                mps_err.error("Items remaining in set are ", *region);
            }
#endif
            region_type* region = nullptr;
            while (free_list_.pop(region)) {
                // clear our stack, we don't need to delete the regions
                // because they are only copies of pointers held by
                // the region_list_
            }

            // delete the regions - better to delete them here than when clearing
            // the stack above in case some were not released by the user
            for (auto r : region_list_) {
                delete r;
            }
            region_list_.clear();

            // release references to shared arrays
            block_list_.clear();
        }

        // ------------------------------------------------------------------------
        inline void push(region_type *region)
        {
#ifdef RMA_POOL_DEBUG_SET
            {
                std::lock_guard<std::mutex> l(set_mutex_);
                region_set_.erase(region);
            }
#endif
            GHEX_DP_ONLY(mps_deb, trace(hpx::debug::str<>(PoolType::desc()), "Push block", *region
               , "Used", hpx::debug::dec<>(in_use_-1)
               , "Accesses", hpx::debug::dec<>(accesses_)));

            if (mps_deb.is_enabled()) {
                uintptr_t val = uintptr_t(region->get_address());
                GHEX_DP_ONLY(mps_deb, trace(hpx::debug::str<>(PoolType::desc())
                   , "Writing 0xdeadbeef to region address"
                   , hpx::debug::ptr(val)));
                if (region->get_address()!=nullptr) {
                    // get use the pointer to the region
                    uintptr_t *ptr = reinterpret_cast<uintptr_t*>(val);
                    for (unsigned int c=0; c<ChunkSize/8; ++c) {
                        ptr[c] = 0xdeadbeef;
                    }
                }
            }

            if (!free_list_.push(region)) {
                mps_err.error(PoolType::desc(), "Error in memory pool push"
                   , *region);
            }
            // decrement one reference
            --in_use_;
        }

        // ------------------------------------------------------------------------
        inline region_type *pop()
        {
            // get a block
            region_type *region = nullptr;
            if (!free_list_.pop(region)) {
                GHEX_DP_ONLY(mps_deb, error(hpx::debug::str<>(PoolType::desc())
                    , "Retry : memory pool pop - increasing allocation"));
                // we must allocate some more memory
                allocate_pool(in_use_);
                return nullptr;
            }
            ++in_use_;
            ++accesses_;
            GHEX_DP_ONLY(mps_deb, trace(hpx::debug::str<>(PoolType::desc()), "Pop block"
               , *region
               , "Used", hpx::debug::dec<>(in_use_)
               , "Accesses", hpx::debug::dec<>(accesses_)));

#ifdef RMA_POOL_DEBUG_SET
            {
                std::lock_guard<std::mutex> l(set_mutex_);
                region_set_.insert(region);
            }
#endif
            return region;
        }

        // ------------------------------------------------------------------------
        // at shutdown we might want to disregrard any bocks still preposted as
        // we can't unpost them
        void decrement_used_count(uint32_t N) {
            in_use_ -= N;
        }

        // ------------------------------------------------------------------------
        // for debug log messages
        std::string status() {
            std::stringstream temp;
            temp << "| " << PoolType::desc()
                 << " ChunkSize " << hpx::debug::hex<6>(ChunkSize)
                 << " Free " << hpx::debug::dec<>(chunks_avail_-in_use_)
                 << " Used " << hpx::debug::dec<>(in_use_)
                 << " Accesses " << hpx::debug::dec<>(accesses_);
            return temp.str();
        }

        // ------------------------------------------------------------------------
        constexpr std::size_t chunk_size() const { return ChunkSize; }

        // ------------------------------------------------------------------------
        // these are counters used for debugging that are usually optimized out
        performance_counter<unsigned int>                 accesses_;
        performance_counter<unsigned int>                 in_use_;
        performance_counter<unsigned int>                 chunks_avail_;
        //
        domain_type                                      *pd_;
        std::unordered_map<const char *, region_ptr>      block_list_;
        std::vector<region_type*>                         region_list_;
        // pool is dynamically sized and can grow if needed
        bl::stack<region_type*, bl::fixed_sized<false>>   free_list_;

#ifdef RMA_POOL_DEBUG_SET
        std::mutex             set_mutex_;
        std::set<region_type*> region_set_;
#endif
    };

}}}}}}
