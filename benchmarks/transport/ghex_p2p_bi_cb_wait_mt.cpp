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
#include <iostream>
#include <vector>
#include <atomic>
#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <ghex/common/timer.hpp>
#include <ghex/transport_layer/util/barrier.hpp>
#include "utils.hpp"

namespace ghex = gridtools::ghex;

#ifdef USE_UCP
// UCX backend
#include <ghex/transport_layer/ucx/context.hpp>
using transport    = ghex::tl::ucx_tag;

#elif defined(USE_LIBFABRIC)
// libfabric backend
#include <ghex/transport_layer/libfabric/context.hpp>
using transport    = ghex::tl::libfabric_tag;

#else
// MPI backend
#include <ghex/transport_layer/mpi/context.hpp>
using transport    = ghex::tl::mpi_tag;
#endif

#ifndef LIBFABRIC_PROGRESS_STRING
# define LIBFABRIC_PROGRESS_STRING "manual"
#endif

#ifndef LIBFABRIC_ENDPOINT_STRING
# define LIBFABRIC_ENDPOINT_STRING "single"
#endif
const char *syncmode = "callback";
const char *waitmode = "wait";

#include <ghex/transport_layer/shared_message_buffer.hpp>
using context_type = typename ghex::tl::context_factory<transport>::context_type;
using communicator_type = typename context_type::communicator_type;
using future_type = typename communicator_type::request_cb_type;
using allocator_type = typename communicator_type::template allocator_type<unsigned char>;
using MsgType = gridtools::ghex::tl::shared_message_buffer<allocator_type>;
using tag_type = typename communicator_type::tag_type;


#ifdef USE_OPENMP
std::atomic<int> sent(0);
std::atomic<int> received(0);
#else
int sent;
int received;
#endif

#ifdef USE_OPENMP
#define THREADID omp_get_thread_num()
#else
#define THREADID 0
#endif

int main(int argc, char *argv[])
{
    int niter, buff_size;
    int inflight;
    int mode;
    gridtools::ghex::timer timer, ttimer;

    if(argc != 4)
    {
        std::cerr << "Usage: bench [niter] [msg_size] [inflight]" << "\n";
        std::terminate();
    }
    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);

    int num_threads = 1;
    gridtools::ghex::tl::barrier_t barrier;
#ifdef USE_OPENMP
#pragma omp parallel
    {
#pragma omp master
        num_threads = omp_get_num_threads();
    }
#endif

#ifdef USE_OPENMP
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mode);
    if(mode != MPI_THREAD_MULTIPLE){
        std::cerr << "MPI_THREAD_MULTIPLE not supported by MPI, aborting\n";
        std::terminate();
    }
#pragma omp parallel
    {
#pragma omp master
        num_threads = omp_get_num_threads();
    }
#else
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &mode);
#endif

    {
        auto context_ptr = ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
        auto& context = *context_ptr;

#ifdef USE_OPENMP
#pragma omp parallel
#endif
        {
            auto comm              = context.get_communicator();
            const auto rank        = comm.rank();
            const auto size        = comm.size();
            const auto thread_id   = THREADID;
            const auto peer_rank   = (rank+1)%2;

            bool using_mt = false;
#ifdef USE_OPENMP
            using_mt = true;
#endif

            int comm_cnt = 0, nlsend_cnt = 0, nlrecv_cnt = 0;

            auto send_callback = [&](communicator_type::message_type, int, tag_type tag)
                 {
                     // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
                     int pthr = tag/inflight;
                     if(pthr != thread_id) nlsend_cnt++;
                     comm_cnt++;
                     sent++;
                 };

            auto recv_callback = [&](communicator_type::message_type, int, tag_type tag)
                 {
                     // std::cout << "recv callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
                     int pthr = tag/inflight;
                     if(pthr != thread_id) nlrecv_cnt++;
                     comm_cnt++;
                     received++;
                 };

            if (thread_id==0 && rank==0)
        {
            if(rank==0)     std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << typeid(comm).name() << "\n\n";
        }

            std::vector<MsgType> smsgs;
            std::vector<MsgType> rmsgs;
            std::vector<future_type> sreqs;
            std::vector<future_type> rreqs;

            for(int j=0; j<inflight; j++){
                smsgs.emplace_back(buff_size);
                rmsgs.emplace_back(buff_size);
                make_zero(smsgs[j]);
                make_zero(rmsgs[j]);
            }
            sreqs.resize(inflight);
            rreqs.resize(inflight);

#ifdef USE_OPENMP
#pragma omp single
#endif
            barrier.rank_barrier(comm);
#ifdef USE_OPENMP
#pragma omp barrier
#endif

            if (thread_id == 0)
        {
            timer.tic();
            ttimer.tic();
            if(rank == 1)
            std::cout << "number of threads: " << num_threads << ", multi-threaded: " << using_mt << "\n";
        }

            // send / recv niter messages, work in inflight requests at a time
            int i = 0, dbg = 0;
            int last_i = 0;
            while(i<niter){

#ifdef USE_OPENMP
#pragma omp barrier
#endif
                if(thread_id == 0 && dbg >= (niter/10)) {
                    dbg = 0;
                    std::cout << rank << " total bwdt MB/s:      "
                              << ((double)(i-last_i)*size*buff_size)/timer.stoc()
                              << "\n";
                    timer.tic();
                    last_i = i;
                }

                // submit inflight requests
                for(int j=0; j<inflight; j++){
                    dbg+=num_threads;
                    i+=num_threads;
                    rreqs[j] = comm.recv(rmsgs[j], peer_rank, thread_id*inflight+j, recv_callback);
                    sreqs[j] = comm.send(smsgs[j], peer_rank, thread_id*inflight+j, send_callback);
                }

                // complete all inflight requests before moving on
                while(sent < num_threads*inflight || received < num_threads*inflight){
                    comm.progress();
                }

#ifdef USE_OPENMP
#pragma omp barrier
#endif
                sent = 0;
                received = 0;
            }

#ifdef USE_OPENMP
#pragma omp single
#endif
            barrier.rank_barrier(comm);
#ifdef USE_OPENMP
#pragma omp barrier
#endif
            if(thread_id==0 && rank == 0)
        {
            const auto t = ttimer.stoc();
            double bw = ((double)niter*size*buff_size)/t;
            // clang-format off
            std::cout << "time:       " << t/1000000 << "s\n";
            std::cout << "final MB/s: " << bw << "\n";
            std::cout << "CSVData"
                      << ", niter, " << niter
                      << ", buff_size, " << buff_size
                      << ", inflight, " << inflight
                      << ", num_threads, " << num_threads
                      << ", syncmode, " << syncmode
                      << ", waitmode, " << waitmode
                      << ", transport, " << ghex::tl::tag_to_string(transport{})
                      << ", BW MB/s, " << bw
                      << ", progress, " << LIBFABRIC_PROGRESS_STRING
                      << ", endpoint, " << LIBFABRIC_ENDPOINT_STRING
                      << "\n";
            // clang-format on
        }

            // stop here to help produce a nice std output
#ifdef USE_OPENMP
#pragma omp single
#endif
            barrier.rank_barrier(comm);
#ifdef USE_OPENMP
#pragma omp barrier
#endif

#ifdef USE_OPENMP
#pragma omp critical
#endif
            {
                std::cout
                    << "rank " << rank << " thread " << thread_id << " serviced " << comm_cnt
                    << ", non-local sends " << nlsend_cnt << " non-local recvs " << nlrecv_cnt << "\n";
            }

            // tail loops - not needed in wait benchmarks
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
