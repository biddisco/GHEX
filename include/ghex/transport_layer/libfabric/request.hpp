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
#include <ghex/threads/atomic/primitives.hpp>
#include <ghex/transport_layer/context.hpp>
#include <ghex/transport_layer/callback_utils.hpp>
#include <ghex/transport_layer/libfabric/controller.hpp>

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace libfabric {

            /** @brief the type of the communication */
            enum class request_kind : int { none=0, send, recv };

            /** @brief simple holder for a shared bool (ready flag) */
            struct request_t
            {
                using controller_type = ghex::tl::libfabric::controller;
                controller_type *m_controller = nullptr;
                request_kind m_kind           = request_kind::none;
                std::shared_ptr<bool> m_ready = nullptr;

                request_t() noexcept = default;

                // we don't bother checking if m_controller is valid because
                // a request should never be created without it
                bool test()
                {
                    if (!*m_ready) {
                        m_controller->poll_for_work_completions();
                    }
                    return *m_ready;
                }

                void wait()
                {
                    while (!test());
                }

                bool cancel()
                {
                    // @TODO not yet implemented
                    return true;
                }

            };

            } // namespace libfabric
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_LIBFABRIC_REQUEST_HPP */
