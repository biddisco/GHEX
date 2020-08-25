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

            // This struct holds the ready state of a future
            // we must also store the context used in libfabric, in case
            // a request is cancelled - fi_cancel(...) needs it
            struct context_info {
                bool  m_ready;
                ghex::tl::libfabric::receiver *m_lf_context;
            };

            /** @brief simple holder for a shared bool (ready flag) */
            struct request_t
            {
                using controller_type = ghex::tl::libfabric::controller;
                controller_type *m_controller           = nullptr;
                request_kind m_kind                     = request_kind::none;
                std::shared_ptr<context_info> m_lf_ctxt = nullptr;

                request_t() noexcept = default;

                // we don't bother checking if m_controller is valid because
                // a request should never be created without it
                bool test()
                {
                    if (!m_lf_ctxt ) {
                        return true;
                    }
                    if (!m_lf_ctxt->m_ready) {
                        m_controller->poll_for_work_completions();
                    }
                    return m_lf_ctxt->m_ready;
                }

                void wait()
                {
                    while (!test());
                }

                bool cancel()
                {
                    // we can  only cancel recv requests...
                    if  (m_kind == request_kind::recv && (m_lf_ctxt->m_lf_context!=nullptr)) {
                        return m_lf_ctxt->m_lf_context->cancel();
                    }
                    return false;
                }

            };

            } // namespace libfabric
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_LIBFABRIC_REQUEST_HPP */
