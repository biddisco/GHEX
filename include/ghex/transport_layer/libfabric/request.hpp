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
#ifndef INCLUDED_GHEX_TL_LIBFABRIC_REQUEST_HPP
#define INCLUDED_GHEX_TL_LIBFABRIC_REQUEST_HPP

#include <functional>
#include "../context.hpp"
#include "../callback_utils.hpp"
#include "../../threads/atomic/primitives.hpp"
#include <ghex/transport_layer/libfabric/controller.hpp>

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace libfabric {

            /** @brief the type of the communication */
            enum class request_kind : int { none=0, send, recv };

            /** @brief thin wrapper around MPI_Request */
            struct request_t
            {
                using controller_type = ::ghex::tl::libfabric::controller;
                controller_type *m_controller = nullptr;
                request_kind m_kind           = request_kind::none;
                std::unique_ptr<bool> m_ready = nullptr;

                request_t() noexcept = default;

//                request_t(controller_type *cont, ) noexcept
//                    : m_controller(cont)
//                    , m_kind(request_kind::none)
//                    , m_ready(false) {}

                bool test()
                {
//                    if (!m_controller) return true;
                    if (!*m_ready) {
                        m_controller->poll_for_work_completions();
                    }
                    return *m_ready;
                }

                void wait()
                {
                    if (!m_controller) return;
                    while (!test());
                }

                bool cancel()
                {
                    // @TODO not yet implemented
                }

            };

            } // namespace libfabric
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_LIBFABRIC_REQUEST_HPP */
