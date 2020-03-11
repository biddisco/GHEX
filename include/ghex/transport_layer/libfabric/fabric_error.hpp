// Copyright (C) 2016 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at

#ifndef HPX_PARCELSET_POLICIES_LIBFABRIC_fabric_error_HPP
#define HPX_PARCELSET_POLICIES_LIBFABRIC_fabric_error_HPP

#include <stdexcept>
#include <string>
#include <string.h>
//
#include <rdma/fi_errno.h>
//
#include <ghex/transport_layer/libfabric/print.hpp>

namespace ghex {
    // cppcheck-suppress ConfigurationNotChecked
    static hpx::debug::enable_print<true> err_deb("ERROR__");
}

namespace ghex {
namespace tl {
namespace libfabric
{

    class fabric_error : public std::runtime_error
    {
    public:
        // --------------------------------------------------------------------
        fabric_error(int err, const std::string &msg)
            : std::runtime_error(std::string(fi_strerror(-err)) + msg),
              error_(err)
        {
            err_deb.error(msg, ":", fi_strerror(-err));
            std::terminate();
        }

        fabric_error(int err)
            : std::runtime_error(fi_strerror(-err)),
              error_(-err)
        {
            err_deb.error(what());
            std::terminate();
        }

        // --------------------------------------------------------------------
        int error_code() const { return error_; }

        // --------------------------------------------------------------------
        static inline char *error_string(int err)
        {
            char buffer[256];
    #if (_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600) && ! _GNU_SOURCE
            return strerror_r(err, buffer, sizeof(buf)) ? nullptr : buffer;
    #else
            return strerror_r(err, buffer, 256);
    #endif
        }

        int error_;
    };

}}}

#endif

