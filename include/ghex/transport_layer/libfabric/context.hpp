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
#ifndef INCLUDED_GHEX_TL_LIBFABRIC_CONTEXT_HPP
#define INCLUDED_GHEX_TL_LIBFABRIC_CONTEXT_HPP

#include "./communicator.hpp"
#include "../communicator.hpp"
#include "./controller.hpp"
#include "./receiver.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {

        // cppcheck-suppress ConfigurationNotChecked
        static hpx::debug::enable_print<false> ctx_deb("CONTEXT");

        template<typename ThreadPrimitives>
        struct transport_context<libfabric_tag, ThreadPrimitives>
        {
            using thread_primitives_type = ThreadPrimitives;
            using communicator_type = communicator<libfabric::communicator<thread_primitives_type>>;
            using thread_token = typename thread_primitives_type::token;
            using shared_state_type = typename communicator_type::shared_state_type;
            using state_type = typename communicator_type::state_type;
            using state_ptr = std::unique_ptr<state_type>;
            using state_vector = std::vector<state_ptr>;

            thread_primitives_type& m_thread_primitives;
            MPI_Comm m_comm;
            std::vector<thread_token>  m_tokens;
            shared_state_type m_shared_state;
            state_type m_state;
            state_vector m_states;
            std::uintptr_t ctag_;

            using controller_type = ghex::tl::libfabric::controller;
            controller_type  *controller_;

            // --------------------------------------------------
            // create a sngleton shared_ptr to a controller that
            // can be shared between ghex context objects
            static controller_type *init_libfabric_controller(int m_rank, int m_size, MPI_Comm mpi_comm) {
                static std::unique_ptr<controller_type> instance(new controller_type(
                        GHEX_LIBFABRIC_PROVIDER,
                        GHEX_LIBFABRIC_DOMAIN,
                        m_rank, m_size, mpi_comm
                    ));
                return instance.get();
            }

            // --------------------------------------------------
            template<typename... Args>
            transport_context(ThreadPrimitives& tp, MPI_Comm mpi_comm, Args&&...)
            : m_thread_primitives(tp)
            , m_comm{mpi_comm}
            , m_tokens(tp.size())
            , m_shared_state(nullptr, this, &tp)
            , m_state(nullptr)
            , m_states(tp.size())
            {
                int m_rank{ [](MPI_Comm c){ int r; GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(c,&r)); return r; }(mpi_comm) };
                int m_size{ [](MPI_Comm c){ int s; GHEX_CHECK_MPI_RESULT(MPI_Comm_size(c,&s)); return s; }(mpi_comm) };

                controller_ = init_libfabric_controller(m_rank, m_size, mpi_comm);
                m_shared_state.m_controller = controller_;
                controller_->startup();
                //
                const int tag_value = 65535;
                if (m_rank==0) {
                    ctag_ = reinterpret_cast<std::uintptr_t>(this);
                    for (int i=1; i<m_size; ++i) {
                        MPI_Send(&ctag_, sizeof(std::uintptr_t), MPI_CHAR, i, tag_value, m_comm);
                    }
                }
                else {
                    MPI_Status status;
                    MPI_Recv(&ctag_, sizeof(std::uintptr_t), MPI_CHAR, 0, tag_value, m_comm, &status);
                }
            }

            communicator_type get_serial_communicator()
            {
                return {&m_shared_state, &m_state};
            }

            communicator_type get_communicator(const thread_token& t)
            {
                if (!m_states[t.id()])
                {
                    m_tokens[t.id()] = t;
                    m_states[t.id()] = std::make_unique<state_type>(&m_tokens[t.id()]);
                }
                return {&m_shared_state, m_states[t.id()].get()};
            }

            controller_type *get_libfabric_controller()
            {
                return m_shared_state.m_controller;
            }
        };

        template<class ThreadPrimitives>
        struct context_factory<libfabric_tag, ThreadPrimitives>
        {
            static std::unique_ptr<context<libfabric_tag, ThreadPrimitives>> create(int num_threads, MPI_Comm mpi_comm)
            {
                auto new_comm = detail::clone_mpi_comm(mpi_comm);
                return std::unique_ptr<context<libfabric_tag, ThreadPrimitives>>{
                    new context<libfabric_tag,ThreadPrimitives>{num_threads, new_comm, new_comm}};
            }
        };

        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_LIBFABRIC_CONTEXT_HPP */
