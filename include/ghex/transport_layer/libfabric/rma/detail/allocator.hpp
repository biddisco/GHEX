//  Copyright (c) 2014-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef GHEX_RMA_DETAIL_ALLOCATOR
#define GHEX_RMA_DETAIL_ALLOCATOR

#include "ghex_libfabric_defines.hpp"
#include <ghex/transport_layer/libfabric/rma/atomic_count.hpp>
#include <ghex/transport_layer/libfabric/rma/detail/memory_region_impl.hpp>
#include <ghex/transport_layer/libfabric/rma/detail/memory_region_allocator.hpp>
#include <ghex/transport_layer/libfabric/rma/memory_region.hpp>
#include <ghex/transport_layer/libfabric/rma/memory_pool.hpp>
#include <ghex/transport_layer/libfabric/performance_counter.hpp>
//
#include <boost/core/null_deleter.hpp>
//
#include <atomic>
#include <stack>
#include <unordered_map>
#include <initializer_list>
#include <iostream>
#include <cstddef>
#include <memory>
#include <array>
#include <sstream>
#include <string>

namespace gridtools {
namespace ghex {
namespace tl {
namespace libfabric {
namespace rma
{

    // this is a temporary solution since we cannot use regions
    // this needs to be replaced by an overried of the message buffer itself
    template<typename T>
    struct allocator
    {
        using size_type = std::size_t;
        using value_type = T;
        using traits = std::allocator_traits<allocator<T>>;
        using is_always_equal = std::true_type;

        // --------------------------------------------------
        // create a sngleton shared_ptr to a controller that
        // can be shared between ghex context objects
        using mempool_type = ghex::tl::libfabric::rma::memory_pool<ghex::tl::libfabric::libfabric_region_provider, T>;
        using region_type = typename mempool_type::region_type;

        static mempool_type *mempool_ptr;

        allocator() noexcept {}
        template<typename U>
        allocator(const allocator<U>&) noexcept {}
        template<typename U>
        allocator(allocator<U>&&) noexcept {}

        void init_memory_pool(mempool_type *mempool) noexcept {
            mempool_ptr = mempool;
        }

        [[nodiscard]] T* allocate(size_type n, const void* cvptr = nullptr)
        {
            region_type *region = mempool_ptr->allocate_region(n);
            T* ptr = reinterpret_cast<T*>(region->get_address());
            mempool_ptr->add_address_to_map(ptr, region);
            return ptr;
        }

        // WARN: temporarily leaking memory due to MPICH bug
        void deallocate(T* ptr, size_type n)
        {
            memory_region *region = mempool_ptr->region_from_address(ptr);
            mempool_ptr->remove_address_from_map(ptr, region);
            mempool_ptr->deallocate(dynamic_cast<region_type*>(region));
        }

        void swap(const allocator&) {}

        friend bool operator==(const allocator&, const allocator&) { return true; }
        friend bool operator!=(const allocator&, const allocator&) { return false; }
    };

}}}}}


namespace gridtools {
namespace ghex {
namespace tl {
namespace libfabric {
namespace rma
{
    template<>
    rma::allocator<unsigned char>::mempool_type* allocator<unsigned char>::mempool_ptr = nullptr;
}}}}}

//ghex::tl::libfabric::rma::memory_pool<ghex::tl::libfabric::libfabric_region_provider, unsigned char>
//    ghex::tl::libfabric::rma::allocator<unsigned char>::mempool_ptr = nullptr;

#endif
