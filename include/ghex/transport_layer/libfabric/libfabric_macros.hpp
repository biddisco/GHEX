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

// some legacy code that hasn't been removed used some old macros
// these are just placeholders that make sure things compile

#define HPX_UNUSED(x) (void) x

#ifdef NDEBUG

#define HPX_ASSERT(expr)
#define HPX_ASSERT_MSG(expr, msg)

#else

#define HPX_ASSERT(expr)                                             \
    (!!(expr) ? void() :                                             \
        throw std::runtime_error("Assertion"));

#define HPX_ASSERT_MSG(expr, msg)                                    \
    (!!(expr) ? void() :                                             \
        throw std::runtime_error(msg));

#endif

// clang-format off
#if defined(__GNUC__)
  #define HPX_LIKELY(expr)    __builtin_expect(static_cast<bool>(expr), true)
  #define HPX_UNLIKELY(expr)  __builtin_expect(static_cast<bool>(expr), false)
#else
  #define HPX_LIKELY(expr)    expr
  #define HPX_UNLIKELY(expr)  expr
#endif
// clang-format on
