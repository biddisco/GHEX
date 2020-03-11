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

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace libfabric {

            /** @brief the type of the communication */
            enum class request_kind : int { none=0, send, recv };

            /** @brief thin wrapper around MPI_Request */
            struct request_t
            {
                GHEX_C_STRUCT(req_type, MPI_Request)
                req_type m_req = MPI_REQUEST_NULL;
                request_kind m_kind = request_kind::none;

                void wait()
                {
                    //MPI_Status status;
                    GHEX_CHECK_MPI_RESULT(MPI_Wait(&m_req.get(), MPI_STATUS_IGNORE));
                }

                bool test()
                {
                    //MPI_Status result;
                    int flag = 0;
                    GHEX_CHECK_MPI_RESULT(MPI_Test(&m_req.get(), &flag, MPI_STATUS_IGNORE));
                    return flag != 0;
                }

                operator       MPI_Request&()       noexcept { return m_req; }
                operator const MPI_Request&() const noexcept { return m_req; }
                      MPI_Request& get()       noexcept { return m_req; }
                const MPI_Request& get() const noexcept { return m_req; }
            };

            } // namespace libfabric
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_LIBFABRIC_REQUEST_HPP */
