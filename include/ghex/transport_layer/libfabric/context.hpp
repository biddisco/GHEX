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

#include "ghex/transport_layer/communicator.hpp"
#include "ghex/transport_layer/libfabric/communicator.hpp"
#include "ghex/transport_layer/libfabric/controller.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {
            namespace libfabric {
            // cppcheck-suppress ConfigurationNotChecked
            static hpx::debug::enable_print<false> ctx_deb("CONTEXT");

            using controller_type = ghex::tl::libfabric::controller;

            // --------------------------------------------------
            // create a singleton ptr to a libfabric controller that
            // can be shared between ghex context objects
            static std::shared_ptr<controller_type> init_libfabric_controller(MPI_Comm comm, int rank, int size, int threads)
            {
                // static std::atomic_flag initialized = ATOMIC_FLAG_INIT;
                // if (initialized.test_and_set()) return;

                // only allow one thread to pass, make other wait
                static std::mutex m_init_mutex;
                std::lock_guard<std::mutex> lock(m_init_mutex);
                static std::shared_ptr<controller_type> instance(nullptr);
                if (!instance.get()) {
                    GHEX_DP_ONLY(ctx_deb, debug(hpx::debug::str<>("New Controller")));
                    instance.reset(new controller_type(
                            GHEX_LIBFABRIC_PROVIDER,
                            GHEX_LIBFABRIC_DOMAIN,
                            comm, rank, size, threads
                        ));
                }
                return instance;
            }

            struct transport_context
            {
                using tag                    = libfabric_tag;
                using communicator_type      = tl::communicator<libfabric::communicator>;
                using shared_state_type      = typename communicator_type::shared_state_type;
                using state_type             = typename communicator_type::state_type;
                using state_ptr              = std::unique_ptr<state_type>;
                using state_vector           = std::vector<state_ptr>;
                using controller_type        = ghex::tl::libfabric::controller;

                std::shared_ptr<controller_type> m_controller;
                state_vector              m_states;
                shared_state_type         m_shared_state;
                state_type                m_state;
                std::mutex                m_mutex;

                // --------------------------------------------------
                transport_context(const mpi::rank_topology& t, int threads)
                    : m_controller{nullptr}
                    , m_states()
                    , m_shared_state{t}
                    , m_state{nullptr}
                    , m_mutex()
                {
                    int rank, size;
                    GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(t.mpi_comm(), &rank));
                    GHEX_CHECK_MPI_RESULT(MPI_Comm_size(t.mpi_comm(), &size));
                    m_controller = init_libfabric_controller(t.mpi_comm(), rank, size, threads);
                    //
                    m_shared_state.init(m_controller.get());
                    m_state.init(m_controller.get());
                }

                communicator_type get_serial_communicator()
                {
                    return {&m_shared_state, &m_state};
                }

                communicator_type get_communicator()
                {
                    // we need to guard only the insertion in the vector,
                    // but this is not a performance critical section
                    std::lock_guard<std::mutex> lock(m_mutex);
                    //
                    m_states.push_back(std::make_unique<state_type>(m_controller->get_tx_endpoint()));
                    return {&m_shared_state, m_states.back().get()};
                }

                controller_type *get_libfabric_controller()
                {
                    return m_controller.get();
                }
            };

        } // namespace libfabric

        template<>
        struct context_factory<libfabric_tag>
        {
            using context_type = context<libfabric::transport_context>;
            static std::unique_ptr<context_type> create(MPI_Comm mpi_comm)
            {
                auto new_comm = detail::clone_mpi_comm(mpi_comm);
                int threads = 1;
#ifdef USE_OPENMP
#pragma omp parallel
    {
#pragma omp master
        threads = omp_get_num_threads();
    }
#endif
                return std::unique_ptr<context_type>{
                    new context_type(new_comm, threads)};
            }
        };

        } // namespace tl
    } // namespace ghex
} // namespace gridtools
