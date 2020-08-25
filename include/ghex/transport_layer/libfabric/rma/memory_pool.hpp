//  Copyright (c) 2014-2017 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef GHEX_RMA_MEMORY_POOL
#define GHEX_RMA_MEMORY_POOL

#include "ghex_libfabric_defines.hpp"
//
#include "atomic_count.hpp"
//
#include "detail/memory_region_impl.hpp"
#include "detail/memory_block_allocator.hpp"
#include "detail/memory_pool_stack.hpp"
//
#include <ghex/transport_layer/libfabric/performance_counter.hpp>
#include <ghex/transport_layer/libfabric/unordered_map.hpp>
//
#include <boost/lockfree/stack.hpp>
//
#include <atomic>
#include <stack>
#include <unordered_map>
#include <iostream>
#include <cstddef>
#include <memory>
#include <array>
#include <sstream>
#include <string>

// the default memory chunk size in bytes
#define RDMA_POOL_1K_CHUNK_SIZE     0x001*0x0400 //  1KB
#define RDMA_POOL_SMALL_CHUNK_SIZE  0x010*0x0400 // 16KB
#define RDMA_POOL_MEDIUM_CHUNK_SIZE 0x040*0x0400 // 64KB
#define RDMA_POOL_LARGE_CHUNK_SIZE  0x400*0x0400 //  1MB

#define RDMA_POOL_MAX_1K_CHUNKS     1024
#define RDMA_POOL_MAX_SMALL_CHUNKS  2048
#define RDMA_POOL_MAX_MEDIUM_CHUNKS 256
#define RDMA_POOL_MAX_LARGE_CHUNKS  256

// if the HPX configuration has set a different value, use it
#if defined(GHEX_LIBFABRIC_MEMORY_CHUNK_SIZE)
# undef RDMA_POOL_SMALL_CHUNK_SIZE
# define RDMA_POOL_SMALL_CHUNK_SIZE GHEX_LIBFABRIC_MEMORY_CHUNK_SIZE
#else
# define GHEX_LIBFABRIC_MEMORY_CHUNK_SIZE 1024
# define RDMA_POOL_SMALL_CHUNK_SIZE GHEX_LIBFABRIC_MEMORY_CHUNK_SIZE
#endif

static_assert ( GHEX_LIBFABRIC_MEMORY_CHUNK_SIZE<RDMA_POOL_MEDIUM_CHUNK_SIZE ,
"Default memory Chunk size must be less than medium chunk size" );

namespace gridtools { namespace ghex {
    // cppcheck-suppress ConfigurationNotChecked
    static hpx::debug::enable_print<false> pool_deb("MEMPOOL");
}}

namespace gridtools {
namespace ghex {
namespace tl {
namespace libfabric {
namespace rma
{

    //----------------------------------------------------------------------------
    // memory pool base class we need so that STL compatible allocate/deallocate
    // routines can be piggybacked onto our registered memory pool API using an
    // abstract allocator interface.
    // For performance reasons - The only member functions that are declared virtual
    // are the STL compatible ones that are only used by the rma_object API
    //----------------------------------------------------------------------------
    struct memory_pool_base
    {
        virtual ~memory_pool_base() {}

        //----------------------------------------------------------------------------
        // release a region back to the pool
        virtual void release_region(memory_region *region) = 0;

        //----------------------------------------------------------------------------
        virtual memory_region *get_region(size_t length) = 0;
    };

    // ---------------------------------------------------------------------------
    // The memory pool manages a collection of memory stacks, each one of which
    // contains blocks of memory of a fixed size. The memory pool holds 4
    // stacks and gives them out in response to allocation requests.
    // Individual blocks are pushed/popped to the stack of the right size
    // for the requested data
    // ---------------------------------------------------------------------------
    template <typename RegionProvider, typename T = unsigned char>
    struct memory_pool : memory_pool_base
    {
        HPX_NON_COPYABLE(memory_pool);

        using mem_pool_element_type = T;

        typedef typename RegionProvider::provider_domain        domain_type;
        typedef detail::memory_region_impl<RegionProvider>      region_type;
        typedef detail::memory_block_allocator<RegionProvider> allocator_type;
        typedef std::shared_ptr<region_type>                    region_ptr;

        //----------------------------------------------------------------------------
        // constructor
        memory_pool(domain_type *pd) :
            protection_domain_(pd),
            tiny_  (pd),
            small_ (pd),
            medium_(pd),
            large_ (pd),
            temp_regions(0),
            user_regions(0)
        {
            tiny_.allocate_pool();
            small_.allocate_pool();
            medium_.allocate_pool();
            large_.allocate_pool();
            pool_deb.debug(hpx::debug::str<>("initialization"), "complete");
        }

        //----------------------------------------------------------------------------
        // destructor
        ~memory_pool()
        {
            deallocate_pools();
        }

        //----------------------------------------------------------------------------
        void deallocate_pools()
        {
            tiny_.DeallocatePool();
            small_.DeallocatePool();
            medium_.DeallocatePool();
            large_.DeallocatePool();
        }

        //----------------------------------------------------------------------------
        // query the pool for a chunk of a given size to see if one is available
        // this function is 'unsafe' because it is not thread safe and another
        // thread may push/pop a block after/during this call and invalidate the result.
        bool can_allocate_unsafe(size_t length) const
        {
            if (length<=tiny_.chunk_size()) {
                return !tiny_.free_list_.empty();
            }
            else if (length<=small_.chunk_size()) {
                return !small_.free_list_.empty();
            }
            else if (length<=medium_.chunk_size()) {
                return !medium_.free_list_.empty();
            }
            else if (length<=large_.chunk_size()) {
                return !large_.free_list_.empty();
            }
            return true;
        }

        //----------------------------------------------------------------------------
        // allocate a region, if size=0 a tiny region is returned
        region_type *allocate_region(size_t length)
        {
            region_type *region = nullptr;
            //
            if (length<=tiny_.chunk_size()) {
                region = tiny_.pop();
            }
            else if (length<=small_.chunk_size()) {
                region = small_.pop();
            }
            else if (length<=medium_.chunk_size()) {
                region = medium_.pop();
            }
            else if (length<=large_.chunk_size()) {
                region = large_.pop();
            }
            // if we didn't get a block from the cache, create one on the fly
            if (region==nullptr) {
                region = allocate_temporary_region(length);
            }

            pool_deb.trace(hpx::debug::str<>("Popping Block")
                , *region
                , tiny_.status()
                , small_.status()
                , medium_.status()
                , large_.status()
                , large_.status()
                , "temp regions" , hpx::debug::dec<>(temp_regions));

            return region;
        }

        //----------------------------------------------------------------------------
        // release a region back to the pool
        void deallocate(region_type *region)
        {
            // if this region was registered on the fly, then don't return it to the pool
            if (region->get_temp_region() || region->get_user_region()) {
                if (region->get_temp_region()) {
                    --temp_regions;
                    pool_deb.trace(hpx::debug::str<>("Deallocating")
                        , "temp region", *region
                        , "temp regions" , hpx::debug::dec<>(temp_regions));
                }
                else if (region->get_user_region()) {
                    --user_regions;
                    pool_deb.trace(hpx::debug::str<>("Deleting")
                        , "(user region)", *region
                        , "user regions" , hpx::debug::dec<>(user_regions));
                }
                delete region;
                return;
            }

            // put the block back on the free list
            if (region->get_size()<=tiny_.chunk_size()) {
                tiny_.push(region);
            }
            else if (region->get_size()<=small_.chunk_size()) {
                small_.push(region);
            }
            else if (region->get_size()<=medium_.chunk_size()) {
                medium_.push(region);
            }
            else if (region->get_size()<=large_.chunk_size()) {
                large_.push(region);
            }

            pool_deb.trace(hpx::debug::str<>("Pushing Block")
                , *region
                , tiny_.status()
                , small_.status()
                , medium_.status()
                , large_.status()
                , "temp regions" , hpx::debug::dec<>(temp_regions));

        }

        //----------------------------------------------------------------------------
        // allocates a region from the heap and registers it, it bypasses the pool
        // when deallocted, it will be unregistered and deleted, not returned to the pool
        region_type* allocate_temporary_region(std::size_t length)
        {
            region_type *region = new region_type();
            region->set_temp_region();
            region->allocate(protection_domain_, length);
            ++temp_regions;
            pool_deb.trace(hpx::debug::str<>("Allocating")
                , "temp region", *region
                , "temp regions" , hpx::debug::dec<>(temp_regions));
            return region;
        }

        //----------------------------------------------------------------------------
        // registers a user allocated address and returns a region,
        // it will be unregistered and deleted, not returned to the pool
        region_type* register_temporary_region(const void *ptr, std::size_t length)
        {
            region_type *region = new region_type(protection_domain_, ptr, length);
            region->set_temp_region();
            ++temp_regions;
            pool_deb.trace(hpx::debug::str<>("Registered")
                , "temp region", *region
                , "temp regions" , hpx::debug::dec<>(temp_regions));
            return region;
        }

        void release_region(memory_region *region) override
        {
            deallocate(dynamic_cast<region_type*>(region));
        }

        //----------------------------------------------------------------------------
        // provide only to allow base class to return a region, without making main
        // region allocate virtual for normal use
        memory_region *get_region(size_t length) {
            return allocate_region(length);
        }

        //----------------------------------------------------------------------------
        // find a memory_region* from the memory address it wraps
        // DEPRECATED : Left for posible reuse
        // this is only valid for regions allocated using the STL allocate method
        // and if you are using this, then you probably should have allocated
        // a region instead in the first place
        memory_region *region_from_address(void const * addr)
        {
            pool_deb.trace(hpx::debug::str<>("Expensive")
                , "region_from_address");
            auto present = region_alloc_pointer_map_.is_in_map(addr);
            if (present.second) {
                pool_deb.trace(hpx::debug::str<>("Found in map")
                    , hpx::debug::ptr(addr)
                    , *(present.first->second));
                return (present.first->second);
            }
            pool_deb.error(hpx::debug::str<>("Not found in map")
                           , hpx::debug::ptr(addr)
                           , "region_from_address returning nullptr");
            return nullptr;
        }

        void add_address_to_map(void const * addr, region_type* region)
        {
            pool_deb.trace(hpx::debug::str<>("Expensive")
                , "add_address_to_map"
                , hpx::debug::ptr(addr), *region);
            std::pair<void const *, region_type*> val(addr, region);
            region_alloc_pointer_map_.insert(val);
        }

        void remove_address_from_map(void const * addr, memory_region* region)
        {
            pool_deb.trace(hpx::debug::str<>("Expensive")
                , "remove_address_from_map"
                , hpx::debug::ptr(addr), *region);
            region_alloc_pointer_map_.erase(addr);
        }

        //----------------------------------------------------------------------------
        // protection domain that memory is registered with
        domain_type *protection_domain_;

        // maintain 4 pools of thread safe pre-allocated regions of fixed size.
        detail::memory_pool_stack<RegionProvider, allocator_type, detail::pool_tiny,
            RDMA_POOL_1K_CHUNK_SIZE,         RDMA_POOL_MAX_1K_CHUNKS> tiny_;
        detail::memory_pool_stack<RegionProvider, allocator_type, detail::pool_small,
            RDMA_POOL_SMALL_CHUNK_SIZE,   RDMA_POOL_MAX_SMALL_CHUNKS> small_;
        detail::memory_pool_stack<RegionProvider, allocator_type, detail::pool_medium,
            RDMA_POOL_MEDIUM_CHUNK_SIZE, RDMA_POOL_MAX_MEDIUM_CHUNKS> medium_;
        detail::memory_pool_stack<RegionProvider, allocator_type, detail::pool_large,
            RDMA_POOL_LARGE_CHUNK_SIZE,   RDMA_POOL_MAX_LARGE_CHUNKS> large_;

        // counters
        hpx::util::atomic_count temp_regions;
        hpx::util::atomic_count user_regions;

        //----------------------------------------------------------------------------
        // used to map the internal memory address to the region that
        // holds the registration information
        hpx::concurrent::unordered_map<const void *, region_type*>
            region_alloc_pointer_map_;
    };

}}}}}

#endif
