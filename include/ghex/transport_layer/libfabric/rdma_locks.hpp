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

#define HPX_PARCELPORT_LIBFABRIC_DEBUG_LOCKS

#include <ghex/transport_layer/libfabric/print.hpp>

#include <mutex>
#include <shared_mutex>

static hpx::debug::enable_print<false> rdma_deb("RMALOCK");

namespace gridtools {
namespace ghex {
namespace tl {
namespace libfabric
{
#ifdef HPX_PARCELPORT_LIBFABRIC_DEBUG_LOCKS
    template<typename Mutex>
    struct scoped_lock: std::lock_guard<Mutex>
    {
        scoped_lock(Mutex &m) : std::lock_guard<Mutex>(m)
        {
            GHEX_DP_ONLY(rdma_deb, debug(hpx::debug::str<>("Creating"), "scoped_lock RAII"));
        }

        ~scoped_lock()
        {
            GHEX_DP_ONLY(rdma_deb, debug(hpx::debug::str<>("Destroying"), "scoped_lock RAII"));
        }
    };

    template<typename Mutex>
    struct unique_lock: std::unique_lock<Mutex>
    {
        unique_lock(Mutex &m) : std::unique_lock<Mutex>(m)
        {
            GHEX_DP_ONLY(rdma_deb, debug(hpx::debug::str<>("Creating"), "unique_lock RAII"));
        }

        unique_lock(Mutex& m, std::try_to_lock_t t) : std::unique_lock<Mutex>(m, t)
        {
            GHEX_DP_ONLY(rdma_deb, debug(hpx::debug::str<>("Creating"), "unique_lock try_to_lock_t RAII"));
        }

        ~unique_lock()
        {
            GHEX_DP_ONLY(rdma_deb, debug(hpx::debug::str<>("Destroying"), "unique_lock RAII"));
        }
    };

    template<typename Mutex>
    struct shared_lock: std::shared_lock<Mutex>
    {
        shared_lock(Mutex &m) : std::shared_lock<Mutex>(m)
        {
            GHEX_DP_ONLY(rdma_deb, debug(hpx::debug::str<>("Creating"), "shared_lock RAII"));
        }

        shared_lock(Mutex& m, std::defer_lock_t t) : std::shared_lock<Mutex>(m, t)
        {
            GHEX_DP_ONLY(rdma_deb, debug(hpx::debug::str<>("Creating"), "shared_lock defer_lock_t RAII"));
        }

        shared_lock(Mutex& m, std::try_to_lock_t t) : std::shared_lock<Mutex>(m, t)
        {
            GHEX_DP_ONLY(rdma_deb, debug(hpx::debug::str<>("Creating"), "shared_lock try_to_lock_t RAII"));
        }

        shared_lock(Mutex& m, std::adopt_lock_t t) : std::shared_lock<Mutex>(m, t)
        {
            GHEX_DP_ONLY(rdma_deb, debug(hpx::debug::str<>("Creating"), "shared_lock adopt_lock_t RAII"));
        }

        ~shared_lock()
        {
            GHEX_DP_ONLY(rdma_deb, debug(hpx::debug::str<>("Destroying"), "shared_lock RAII"));
        }
    };
#else
    template<typename Mutex>
    using scoped_lock = std::lock_guard<Mutex>;

    template<typename Mutex>
    using unique_lock = std::unique_lock<Mutex>;
#endif
}}}}


