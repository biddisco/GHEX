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
#include <iomanip>
#include <ghex/transport_layer/libfabric/rma/detail/memory_region_allocator.hpp>
#include <ghex/common/timer.hpp>
#include <gtest/gtest.h>

#ifdef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/ucx/context.hpp>
using transport = gridtools::ghex::tl::ucx_tag;
#elif GHEX_TEST_USE_LIBFABRIC
#include <ghex/transport_layer/libfabric/context.hpp>
using transport = gridtools::ghex::tl::libfabric_tag;
#else
#include <ghex/transport_layer/mpi/context.hpp>
using transport = gridtools::ghex::tl::mpi_tag;
#endif

#include <ghex/threads/none/primitives.hpp>
using threading = gridtools::ghex::threads::none::primitives;

template <typename T>
using allocator_type = gridtools::ghex::tl::libfabric::rma::memory_region_allocator<T>;

template <typename T>
using fancy_vector = std::vector<T, allocator_type<T>>;

TEST(transport, barrier) {

    const size_t size = 2;
    allocator_type<double> alloc = allocator_type<double>();
    auto mem = alloc.allocate(size);


    fancy_vector<double> vec;
    vec.resize(size, 0.125);
    for (int i=0; i<size; ++i) {
        vec[i] = (i+1)*1.234;
    }
    std::copy(vec.begin(), vec.end(), std::ostream_iterator<double>(std::cout, ", "));

    auto xxx = vec.front();
    fancy_vector<double>::iterator it = vec.begin();
    std::cout << "Key " << *it << std::endl;
//    it.get_key();
    //std::cout << "Key " << key << std::endl;
    auto key = vec.get_allocator(); //.get_key();


    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    auto token = context.get_token();
    auto comm = context.get_communicator(token);
    int rank = context.rank();
    gridtools::ghex::timer timer;

    timer.tic();
    for(int i=0; i<100; i++)  {
      comm.barrier();
    }
    const auto t = timer.stoc();
    if(rank==0)
    {
        std::cout << "time:       " << t/1000000 << "s\n";
    }
}
