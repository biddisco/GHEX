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
#ifndef INCLUDED_GHEX_TL_LIBFABRIC_ERROR_HPP
#define INCLUDED_GHEX_TL_LIBFABRIC_ERROR_HPP

#include <string>
#include <stdexcept>
//
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include "fabric_error.hpp"

#ifdef NDEBUG
    #define GHEX_CHECK_LIBFABRIC_RESULT(x) x;
#else
    #define GHEX_CHECK_LIBFABRIC_RESULT(x)  \
    if (x) throw fabric_error(x,            \
        "GHEX Error: UCX Call failed " + std::string(#x) + " in " + \
        std::string(__FILE__) + ":" + std::to_string(__LINE__));
#endif

#endif /* INCLUDED_GHEX_TL_LIBFABRIC_ERROR_HPP */

