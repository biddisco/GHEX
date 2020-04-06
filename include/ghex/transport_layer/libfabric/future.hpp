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
#ifndef INCLUDED_GHEX_TL_LIBFABRIC_FUTURE_HPP
#define INCLUDED_GHEX_TL_LIBFABRIC_FUTURE_HPP

#include <ghex/transport_layer/libfabric/request.hpp>

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace libfabric {

            /** @brief future template for non-blocking communication */
            template<typename T>
            struct future_t
            {
                using value_type  = T;
                using handle_type = request_t;

                value_type m_data;
                handle_type m_handle;

                future_t(value_type&& data, handle_type&& h)
                :   m_data(std::move(data))
                ,   m_handle(std::move(h))
                {}

                future_t(const future_t&) = delete;
                future_t(future_t&&) = default;
                future_t& operator=(const future_t&) = delete;
                future_t& operator=(future_t&&) = default;

                void wait() noexcept
                {
                    m_handle.wait();
                }

                bool test() noexcept
                {
                    return m_handle.test();
                }

                bool ready() noexcept
                {
                    return m_handle.test();
                }

                [[nodiscard]] value_type get()
                {
                    wait();
                    return std::move(m_data);
                }

                bool is_recv() const noexcept { return (m_handle.m_kind == request_kind::recv); }

                /** Cancel the future.
                  * @return True if the request was successfully canceled */
                bool cancel()
                {
                    // @TODO not yet implemented
                    return true;
                }
            };

            template<>
            struct future_t<void>
            {
                using handle_type = request_t;

                handle_type m_handle;

                future_t() noexcept = default;
                future_t(handle_type&& h)
                :   m_handle(std::move(h))
                {}
                future_t(const future_t&) = delete;
                future_t(future_t&&) = default;
                future_t& operator=(const future_t&) = delete;
                future_t& operator=(future_t&&) = default;

                void wait() noexcept
                {
                    m_handle.wait();
                }

                bool test() noexcept
                {
                    return m_handle.test();
                }

                bool ready() noexcept
                {
                    return m_handle.test();
                }

                void get()
                {
                    wait();
                }

                bool is_recv() const noexcept { return (m_handle.m_kind == request_kind::recv); }

                bool cancel()
                {
                        return true;
                }
            };

            } // namespace libfabric
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_LIBFABRIC_FUTURE_HPP */

