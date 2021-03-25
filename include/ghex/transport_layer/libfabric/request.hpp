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

#include <functional>
#include <ghex/transport_layer/context.hpp>
#include <ghex/transport_layer/callback_utils.hpp>
#include <ghex/transport_layer/libfabric/controller.hpp>

// cppcheck-suppress ConfigurationNotChecked
static hpx::debug::enable_print<false> req_deb("REQUEST");

namespace gridtools{ namespace ghex { namespace tl { namespace libfabric {

        using region_provider    = libfabric_region_provider;
        using region_type        = rma::detail::memory_region_impl<region_provider>;
        using libfabric_msg_type = message_buffer<rma::memory_region_allocator<unsigned char>>;
        using any_msg_type       = gridtools::ghex::tl::libfabric::any_libfabric_message;

        /** @brief the type of the communication */
        enum class request_kind : int { none=0, send, recv };

        // --------------------------------------------------------------------
        // Extract RMA handle info from the message
        void context_info::init_message_data(const libfabric_msg_type &msg, uint64_t tag)
        {
            tag_ = tag;
            message_region_ = msg.get_buffer().m_pointer.region_;
            message_region_->set_message_length(msg.size());
        }

        void context_info::init_message_data(const any_libfabric_message &msg, uint64_t tag)
        {
            tag_ = tag;
            message_region_ = msg.region();
            message_region_->set_message_length(msg.size());
        }

        template <typename Message,
                  typename Enable = typename std::enable_if<
                      !std::is_same<libfabric_msg_type, Message>::value &&
                      !std::is_same<any_libfabric_message, Message>::value>::type>
        void context_info::init_message_data(Message &msg, uint64_t tag)
        {
            tag_ = tag;
            message_holder_.set_rma_from_pointer(msg.data(), msg.size());
            message_region_ = message_holder_.m_region;
        }

        // --------------------------------------------------------------------
        // Called when a send completes
        int context_info::handle_send_completion()
        {
            GHEX_DP_ONLY(req_deb, debug(hpx::debug::str<>("handle_send_completion")
                                        ,hpx::debug::hex<8>(tag_)));
            // invoke the user supplied callback
            user_cb_();
            return 1;
        }

        int context_info::handle_recv_completion(std::uint64_t /*len*/)
        {
            GHEX_DP_ONLY(req_deb, debug(hpx::debug::str<>("handle_recv_completion")
                                        ,hpx::debug::hex<8>(tag_)));
            // call user supplied completion callback
            user_cb_();
            return 1;
        }


        bool context_info::cancel()
        {
            bool ok = (fi_cancel(&this->endpoint_->get_ep()->fid, this) == 0);
            if (!ok) return ok;

            // cleanup as if we had completed, but without calling user callbacks
            set_ready();
            //user_cb_ = [](){};
            message_region_ = nullptr;
            return ok;
        }


        /** @brief simple holder for a shared bool (ready flag) */
        struct request_t
        {
            using controller_type = ghex::tl::libfabric::controller;
            controller_type *m_controller           = nullptr;
            libfabric::request_kind m_kind          = libfabric::request_kind::none;
            std::shared_ptr<libfabric::context_info> m_lf_ctxt = nullptr;

            request_t() noexcept = default;
            request_t(request_t&&) = default;
            request_t(request_t &) = default;
            request_t& operator= (request_t &&) = default;

            inline bool is_ready()
            {
                return m_lf_ctxt->ready();
            }

            // we don't bother checking if m_controller is valid because
            // a request could/should never be created without it
            bool test()
            {
                if (!m_lf_ctxt ) {
                    return true;
                }
                if (!is_ready()) {
                    m_controller->poll_for_work_completions();
//                    if (m_lf_ctxt->is_send_) {
//                        m_controller->poll_send_queue(m_lf_ctxt->endpoint_->get_tx_cq());
//                    }
//                    else {
//                        m_controller->poll_recv_queue(m_lf_ctxt->endpoint_->get_rx_cq());
//                    }
                }
                return is_ready();
            }

            inline bool ready()
            {
//                return m_lf_ctxt->ready();
                return test();
            }

            void wait()
            {
                while (!test());
            }

            bool cancel()
            {
                // we can only cancel recv requests...
                if  (m_kind == libfabric::request_kind::recv && (m_lf_ctxt!=nullptr)) {
                    return m_lf_ctxt->cancel();
                }
                return false;
            }
        };

}}}}
