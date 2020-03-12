//  Copyright (c) 2014-2017 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_RMA_MEMORY_REGION_ALLOCATOR
#define HPX_PARCELSET_POLICIES_RMA_MEMORY_REGION_ALLOCATOR

#include "ghex_libfabric_defines.hpp"
//
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
#include <array>
#include <sstream>
#include <string>

namespace ghex {
namespace tl {
namespace libfabric {
namespace rma {
namespace detail
{
    // --------------------------------------------------------------------
    // This is a simple class that implements only malloc and free but is
    // templated over the memory region provider which is parcelport
    // dependent. This class is used internally by the rma memory pools to
    // allocate blocks and should not be called by user code directly.
    // The allocator is intended to be used to generate large blocks that are
    // subdivided and used by the memory pools and not to allocate many
    // small blocks. These blocks are returned as shared pointers.
    template <typename RegionProvider>
    struct memory_region_allocator
    {
        typedef typename RegionProvider::provider_domain domain_type;
        typedef memory_region_impl<RegionProvider>       region_type;
        typedef std::shared_ptr<region_type>             region_ptr;

        // default empty constructor
        memory_region_allocator() {}

        // allocate a registered memory region
        static region_ptr malloc(domain_type *pd, const std::size_t bytes)
        {
            region_ptr region = std::make_shared<region_type>();
            LOG_TRACE_MSG("Allocating " << hexuint32(bytes) << "using chunk mallocator");
            region->allocate(pd, bytes);
            return region;
        }

        // release a registered memory region
        static void free(region_ptr region) {
            LOG_TRACE_MSG("Freeing a block from chunk mallocator (ref count) "
                << region.use_count());
            region.reset();
        }
    };

}}}}}

#endif