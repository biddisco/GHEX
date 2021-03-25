/*
 * GridTools
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */
#pragma once

#include <ghex/transport_layer/libfabric/rma/detail/memory_region_traits.hpp>
#include <ghex/transport_layer/libfabric/rma/memory_region.hpp>
#include <ghex/transport_layer/libfabric/print.hpp>
//
#include <memory>

namespace gridtools { namespace ghex {
    // cppcheck-suppress ConfigurationNotChecked
    static hpx::debug::enable_print<false> memr_deb("MEM_REG");
}}

namespace gridtools {
namespace ghex {
namespace tl {
namespace libfabric {
namespace rma {
namespace detail
{
    // --------------------------------------------------------------------
    // a memory region is a pinned block of memory that has been specialized
    // by a particular region provider. Each provider (infiniband, libfabric,
    // other) has a different definition for the region object and the protection
    // domain used to limit access.
    // Code that does not 'know' which transport layer is being used, should use
    // the memory_region class to manage regions, but transport layer code
    // may use the correct type for the layer in question.
    // --------------------------------------------------------------------
    template <typename RegionProvider>
    class memory_region_impl : public memory_region
    {
      public:
        typedef typename RegionProvider::provider_domain provider_domain;
        typedef typename RegionProvider::provider_region provider_region;

        // --------------------------------------------------------------------
        memory_region_impl() :
            memory_region(), region_(nullptr) {}

        // --------------------------------------------------------------------
        memory_region_impl(
            provider_region *region, char *address,
            char *base_address, uint64_t size, uint32_t flags)
            : memory_region(address, base_address, size, flags)
            , region_(region) {}

        // --------------------------------------------------------------------
        // construct a memory region object by registering an existing address buffer
        memory_region_impl(provider_domain *pd, const void *buffer, const uint64_t length)
        {
            address_    = static_cast<char *>(const_cast<void *>(buffer));
            base_addr_  = address_;
            size_       = length;
            used_space_ = length;
            flags_      = BLOCK_USER;

            int ret = traits::rma_memory_region_traits<RegionProvider>::register_memory(
                pd, const_cast<void*>(buffer), length,
                traits::rma_memory_region_traits<RegionProvider>::flags(),
                0, (uint64_t)address_, 0, &(region_), nullptr);

            if (ret) {
                GHEX_DP_ONLY(memr_deb, debug(
                    hpx::debug::str<>("error registering")
                    , hpx::debug::ptr(buffer) , hpx::debug::hex<6>(length)));
            }
            else {
                GHEX_DP_ONLY(memr_deb, trace(
                    hpx::debug::str<>("OK registering")
                    , hpx::debug::ptr(buffer) , hpx::debug::ptr(address_)
                    , "desc" , hpx::debug::ptr(fi_mr_desc(region_))
                    , "rkey" , hpx::debug::ptr(fi_mr_key(region_))
                    , "length" , hpx::debug::hex<6>(size_)));
            }
        }

        // --------------------------------------------------------------------
        // allocate a block of size length and register it
        int allocate(provider_domain *pd, uint64_t length)
        {
            // Allocate storage for the memory region.
            void *buffer = new char[length];
            if (buffer != nullptr) {
                GHEX_DP_ONLY(memr_deb, trace(hpx::debug::str<>("allocated malloc OK")
                    , hpx::debug::hex<6>(length)));
            }
            address_    = static_cast<char*>(buffer);
            base_addr_  = static_cast<char*>(buffer);
            size_       = length;
            used_space_ = 0;

            int ret = traits::rma_memory_region_traits<RegionProvider>::register_memory(
                pd, const_cast<void*>(buffer), length,
                traits::rma_memory_region_traits<RegionProvider>::flags(),
                0, (uint64_t)address_, 0, &(region_), nullptr);

            if (ret) {
                GHEX_DP_ONLY(memr_deb, debug(
                    hpx::debug::str<>("error registering")
                    , hpx::debug::ptr(buffer) , hpx::debug::hex<6>(length)));
            }
            else {
                GHEX_DP_ONLY(memr_deb, trace(
                    hpx::debug::str<>("OK registering")
                    , hpx::debug::ptr(buffer) , hpx::debug::ptr(address_)
                    , "desc" , hpx::debug::ptr(fi_mr_desc(region_))
                    , "rkey" , hpx::debug::ptr(fi_mr_key(region_))
                    , "length" , hpx::debug::hex<6>(size_)));
            }

            GHEX_DP_ONLY(memr_deb, trace(
                  hpx::debug::str<>("memory region") , hpx::debug::ptr(this)
                , *this));
            return 0;
        }

        // --------------------------------------------------------------------
        // destroy the region and memory according to flag settings
        ~memory_region_impl()
        {
            if (get_partial_region()) return;
            release();
        }

        // --------------------------------------------------------------------
        // Deregister and free the memory region.
        // returns 0 when successful, -1 otherwise
        int release(void)
        {
            if (region_ != nullptr) {
                GHEX_DP_ONLY(memr_deb, trace(hpx::debug::str<>("releasing region")
                    , *this));
                // get these before deleting/unregistering (for logging)
                [[maybe_unused]] auto buffer = memr_deb.declare_variable<uint64_t>(get_base_address());
                [[maybe_unused]] auto length = memr_deb.declare_variable<uint64_t>(get_size());
                //
                if (traits::rma_memory_region_traits<RegionProvider>::
                    unregister_memory(region_))
                {
                    GHEX_DP_ONLY(memr_deb, debug("Error, fi_close mr failed\n"));
                    return -1;
                }
                else {
                    GHEX_DP_ONLY(memr_deb, trace(hpx::debug::str<>("deregistered region")
                        , hpx::debug::ptr(get_local_key())
                        , "at address" , hpx::debug::ptr(buffer)
                        , "with length" , hpx::debug::hex<6>(length)));
                }
                if (!get_user_region()) {
                    delete [](static_cast<const char*>(get_base_address()));
                }
                region_ = nullptr;
            }
            return 0;
        }

        // --------------------------------------------------------------------
        // Get the local descriptor of the memory region.
        virtual void* get_local_key(void) const {
            return
                traits::rma_memory_region_traits<RegionProvider>::get_local_key(region_);
        }

        // --------------------------------------------------------------------
        // Get the remote key of the memory region.
        virtual uint64_t get_remote_key(void) const {
            return
                traits::rma_memory_region_traits<RegionProvider>::get_remote_key(region_);
        }

        // --------------------------------------------------------------------
        // return the underlying infiniband region handle
        inline provider_region *get_region() const { return region_; }

    private:
        // The internal network type dependent memory region handle
        provider_region *region_;

    };

}}}}}}
