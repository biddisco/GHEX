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
#ifndef INCLUDED_GHEX_UTILS_HPP
#define INCLUDED_GHEX_UTILS_HPP

#include <sched.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <errno.h>
#include <string.h>

template<typename Msg>
void make_zero(Msg& msg) {
    for (auto& c : msg)
    c = 0;
}

void bind_to_core(int thrid)
{
    cpu_set_t cpu_mask;
    pid_t tid = syscall(SYS_gettid);
    CPU_ZERO(&cpu_mask);
    CPU_SET(thrid, &cpu_mask);
    if (sched_setaffinity(tid, sizeof(cpu_mask), &cpu_mask) == -1){
        fprintf(stderr, "sched_setaffinity error : %s\n", strerror(errno));
        exit(1);
    }
}

#ifndef LIBFABRIC_PROGRESS_STRING
# define LIBFABRIC_PROGRESS_STRING "MANUAL"
#endif

#ifndef LIBFABRIC_ENDPOINT_STRING
    std::string libfabric_endpoint_type()
    {
        auto lf_ep_type = std::getenv("LIBFABRIC_ENDPOINT_TYPE");
        if (lf_ep_type) {
            if (std::string(lf_ep_type)==std::string("threadlocal") || std::atoi(lf_ep_type)==2)
                return "threadlocal";
            if (std::string(lf_ep_type)==std::string("multiple") || std::atoi(lf_ep_type)==1)
                return "multiple";
        }
        return "single";
    }

# define LIBFABRIC_ENDPOINT_STRING libfabric_endpoint_type()
#endif

#endif /* INCLUDED_GHEX_UTILS_HPP */

