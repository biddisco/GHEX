#if !defined(GHEX_LIBFABRIC_MACROS_HPP)
#define GHEX_LIBFABRIC_MACROS_HPP

#define HPX_UNUSED(x) (void) x

#ifndef NDEBUG

#define HPX_ASSERT(expr)
#define HPX_ASSERT_MSG(expr, msg)

#else

#define HPX_ASSERT(expr)                                             \
    (!!(expr) ? void() :                                             \
        throw std::runtime_error("Assertion");

#define HPX_ASSERT_MSG(expr, msg)                                    \
    (!!(expr) ? void() :                                             \
        throw std::runtime_error(msg);

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

#endif
