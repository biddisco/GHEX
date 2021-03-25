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
#include <atomic>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>
#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <ghex/common/timer.hpp>
#include <ghex/transport_layer/util/barrier.hpp>
#include "utils.hpp"

namespace ghex = gridtools::ghex;

// this is for extra debugging of libfabric lockups
// each transport needs to provide a specialization
// mpi/ucx noop/dummy implementations are provided
template<typename Communicator>
struct completions {};

// enable cleaned up debugging output
// clang-format off
#define buffered_out(x) {    \
    std::stringstream temp;  \
    temp << x << std::endl;  \
    std::cout << temp.str(); }
// clang-format on

// uncommment to disable cleaned up debugging output
//#undef buffered_out
//#define buffered_out(x)

std::string print_send_recv_info(std::tuple<int,int,int,int,int,int> &tup) {
    std::stringstream temp;
    temp << " Sends Posted "    << std::get<0>(tup)
         << " Sends Completed " << std::get<1>(tup)
         << " Sends Readied "   << std::get<2>(tup)
         << " Recvs Posted "    << std::get<3>(tup)
         << " Recvs Completed " << std::get<4>(tup)
         << " Recvs Readied "   << std::get<5>(tup)
         << " ";
    return temp.str();
}

#ifdef USE_UCP
// UCX backend
#include <ghex/transport_layer/ucx/context.hpp>
using transport = ghex::tl::ucx_tag;
template <>
struct completions<gridtools::ghex::tl::communicator<gridtools::ghex::tl::ucx::communicator>>
{
    gridtools::ghex::tl::communicator<gridtools::ghex::tl::ucx::communicator> &comm;
    std::string send_recv_info() { return ""; }
    std::tuple<int,int,int,int,int,int> send_recv_data() {
        return std::make_tuple(0,0,0,0,0,0);
    }
};

#elif defined(USE_LIBFABRIC)
// libfabric backend
#include <ghex/transport_layer/libfabric/context.hpp>
using transport = ghex::tl::libfabric_tag;
template <>
struct completions<gridtools::ghex::tl::communicator<gridtools::ghex::tl::libfabric::communicator>>
{
    gridtools::ghex::tl::communicator<gridtools::ghex::tl::libfabric::communicator> &comm;
    std::string send_recv_info() {
        auto tup = comm.get_send_recv_counters();
        return print_send_recv_info(tup);
    }
    std::tuple<int,int,int,int,int,int> send_recv_data() {
        return comm.get_send_recv_counters();
    }
};

#else
// MPI backend
#include <ghex/transport_layer/mpi/context.hpp>
using transport = ghex::tl::mpi_tag;

template <>
struct completions<gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi::communicator>>
{
    gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi::communicator> &comm;
    std::string send_recv_info() { return ""; }
    std::tuple<int,int,int,int,int,int> send_recv_data() {
        return std::make_tuple(0,0,0,0,0,0);
    }
};
#endif

#ifndef LIBFABRIC_PROGRESS_STRING
#define LIBFABRIC_PROGRESS_STRING "manual"
#endif

#ifndef LIBFABRIC_ENDPOINT_STRING
#define LIBFABRIC_ENDPOINT_STRING "single"
#endif
const char* syncmode = "future";
const char* waitmode = "avail";

#include <ghex/transport_layer/message_buffer.hpp>
using context_type =
    typename ghex::tl::context_factory<transport>::context_type;
using communicator_type = typename context_type::communicator_type;
using future_type = typename communicator_type::future<void>;
using allocator_type =
    typename communicator_type::template allocator_type<unsigned char>;
using MsgType = gridtools::ghex::tl::message_buffer<allocator_type>;
using tag_type = typename communicator_type::tag_type;

#ifdef USE_OPENMP
std::atomic<int> sends_posted(0);
std::atomic<int> sends_completed(0);
std::atomic<int> receives_posted(0);
#else
std::atomic<int> sends_posted(0);
std::atomic<int> sends_completed(0);
std::atomic<int> receives_posted(0);
#endif

#ifdef USE_OPENMP
#define THREADID omp_get_thread_num()
#else
#define THREADID 0
#endif

// keep track of sends on a thread local basis
struct msg_tracker {
    std::vector<MsgType> msgs;
    std::vector<future_type> reqs;
    //
    msg_tracker() = default;
    //
    void init(int inflight, int buff_size)
    {
        msgs.resize(inflight);
        reqs.resize(inflight);
        //
        for (int j = 0; j < inflight; j++)
        {
            msgs[j].resize(buff_size);
            make_zero(msgs[j]);
        }
    }
};

int main(int argc, char* argv[])
{
    int buff_size;
    int nsecs;
    int inflight;
    int mode;
    double elapsed;
    gridtools::ghex::timer ttimer;

    if (argc != 4)
    {
        std::cerr << "Usage: bench [nsecs] [msg_size] [inflight]"
                  << "\n";
        std::terminate();
    }
    nsecs = atoi(argv[1]);
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
    if (mode != MPI_THREAD_MULTIPLE)
    {
        std::cerr << "MPI_THREAD_MULTIPLE not supported by MPI, aborting\n";
        std::terminate();
    }
#else
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &mode);
#endif

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    {
        auto context_ptr =
            ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
        auto& context = *context_ptr;

        // @todo : arrange these better to avoid false cache sharing
        msg_tracker sends;
        msg_tracker recvs;
        sends.init(inflight*num_threads, buff_size);
        recvs.init(inflight*num_threads, buff_size);

        // when all is done, we use these to sync msg count between ranks
        MsgType done_send(sizeof(int));
        MsgType done_recv(sizeof(int));
        future_type fsend;
        future_type frecv;
        int num_messages_expected = std::numeric_limits<int>::max()/2;
        std::atomic<int> master_thread(-1);

        // an atomic flag we use to indicate that we have completed
        // sends, and posted a "done sends" message
        std::atomic<int>  threads_completed(0);
        //
        const int num_deb = 5;
        const int msecond = (1000*nsecs)/num_deb;
        //
        auto start    = std::chrono::steady_clock::now();
        auto dbg_time = start;

#ifdef USE_OPENMP
#pragma omp parallel
#endif
        {
            auto comm = context.get_communicator();
            const auto rank = comm.rank();
            const auto size = comm.size();
            const auto thread_id = THREADID;
            const auto peer_rank = (rank + 1) % size;
            bool using_mt = false;
#ifdef USE_OPENMP
            using_mt = true;
#endif

            if (thread_id == 0 && rank == 0)
            {
                std::cout << "\n\nrunning test " << __FILE__
                          << " with communicator " << typeid(comm).name()
                          << "\n\n";
            };

#ifdef USE_OPENMP
#pragma omp single
#endif
            barrier.rank_barrier(comm);
#ifdef USE_OPENMP
#pragma omp barrier
#endif

            if (thread_id == 0) {
                ttimer.tic();
                if (rank == 1)
                    std::cout << "number of threads: " << num_threads
                              << ", multi-threaded: " << using_mt << "\n";
            }

            // NB. these are thread local
            bool thread_sends_complete = false; // true when this thread has completed sends
            bool time_up = false;               // true when time exceeded
            bool thread_once_flag = true;

            // loop for allowed time : sending and receiving
            do {
                auto now = std::chrono::steady_clock::now();
                time_up = (now > start + std::chrono::seconds{nsecs});

                // just one thread does debug printf's
                if (thread_id == 0) {
                    // output debug info at periodic intervals
                    if (now > dbg_time + std::chrono::milliseconds{msecond})
                    {
                        dbg_time = now;
                        completions<decltype(comm)> comp{comm};
                        buffered_out("rank: " << rank << " \tsend: " << sends_posted << "\t recv: " << receives_posted);
//                        buffered_out("rank: " << rank << comp.send_recv_info());
                    }
                }

                for (int j = 0; j < inflight; j++) {
                    int tag = thread_id * inflight + j;
                    if (recvs.reqs[tag].test())
                    {
                        recvs.reqs[tag] = comm.recv(recvs.msgs[tag], peer_rank, tag);
                        receives_posted++;
                    }
                    // if time available and a send slot is available, post a send
                    if (!time_up && sends.reqs[tag].test() && (sends_posted<receives_posted + (2*inflight)))
                    {
                        sends.reqs[tag] = comm.send(sends.msgs[tag], peer_rank, tag);
                        ++sends_posted;
                    }
                }
                // if time is up, keep polling until all send futures are ready
                if (time_up) {
                    // are there any incomplete sends on this thread
                    thread_sends_complete = true;
                    for (int j = 0; j < inflight; j++) {
                        int tag = thread_id * inflight + j;
                        if (!sends.reqs[tag].test())
                            thread_sends_complete = false;
                    }

                    // if this thread has completed its sends
                    if (thread_sends_complete) {
                        // last thread to be ready sends a single "done" message
                        // containing total sent, receive the same from peer
                        if (thread_once_flag) {
                            thread_once_flag = false; // don't re-enter this section
                            // only last thread to finish can trigger this
                            if (++threads_completed == num_threads) {
                                master_thread = thread_id; // we are the master thread
                                buffered_out("rank: " << rank
                                             << " thread " << thread_id
                                             << " bcast SENDS = " << sends_posted);
                                std::memcpy(done_send.data(), &sends_posted, sizeof(int));
                                fsend = comm.send(done_send, peer_rank, 0xffff);
                                frecv = comm.recv(done_recv, peer_rank, 0xffff);
                            }
                            else {
//                                buffered_out("rank: " << rank
//                                             << " thread " << thread_id
//                                             << " thread_sends_complete " << sends_posted);
                            }
                        }
                        // only master thread will check done futures
                        if (thread_id==master_thread) {
                            // our send has completed, and we received peer's
                            if (fsend.test() && frecv.test()) {
                                std::memcpy(&num_messages_expected, done_recv.data(), sizeof(int));
                                buffered_out("rank: " << rank
                                             << " thread " << thread_id
                                             << " expecting " << num_messages_expected
                                             << " need receives " << num_messages_expected + inflight*num_threads
                                             );
                                // don't re-enter this section
                                master_thread = -1;
                            }
                        }
                    }
                }
                // when the number of receives posted is equal to the
                // number of messages sent by the peer + (inflight*num_threads)
                // then all messages sent by them have been received.
            } while (receives_posted != (num_messages_expected + inflight*num_threads));

//            buffered_out("rank: " << rank << " thread " << thread_id << " \tsend: " << sends_posted << "\t recv: " << receives_posted);
        } // end of parallel region

        // all ranks have completed sends : test is over, stop the clock
        // timing includes a few bits of synchronization overhead, but
        // when running for more than a few seconds will be negligable
        elapsed = ttimer.toc();

        // there might be a few sends that are still arriving,
        // loop over receives
        for (int j = 0; j < inflight*num_threads; j++)
        {
            if (!recvs.reqs[j].ready())
            {
                // buffered_out("Cancel extra recv " << j);
                recvs.reqs[j].cancel();
                receives_posted--;
            }
            else
            {
//                throw std::runtime_error("All receive futures should be ready");
                buffered_out("error late receive completion " << j);
            }
        }
        // sanity check, make sure no messages are left
        for (int j = 0; j < inflight*num_threads; j++)
        {
            if (!recvs.reqs[j].ready())
                throw std::runtime_error("Message should have been cancelled");
        }
        if (receives_posted!=num_messages_expected) {
            std::cout << "receives_posted!=num_messages_expected "
                      <<  receives_posted << " "
                       << num_messages_expected << std::endl;
            throw std::runtime_error("Final message count mismatch");
        }

//        buffered_out("rank: " << rank << " Before barrier 3");
        MPI_Barrier(MPI_COMM_WORLD);

        // total traffic is amount sends_posted in both directions
        if (rank == 0)
        {
            double bw = ((double) (sends_posted + receives_posted) * buff_size) / elapsed;
            // clang-format off
            std::cout << "time:       " << elapsed/1000000 << "s\n";
            std::cout << "final MB/s: " << bw << "\n";
            std::cout << "CSVData"
                      << ", niter, " << sends_posted + receives_posted
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
    }
    MPI_Finalize();
}
