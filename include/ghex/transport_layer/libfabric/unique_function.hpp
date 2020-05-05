#ifndef GHEX_LIBFABRIC_UNIQUE_FUNCTION_HPP
#define GHEX_LIBFABRIC_UNIQUE_FUNCTION_HPP

// Temporary work around for move only callbacks

#include <functional>
#include <type_traits>

namespace gridtools {
namespace ghex {
namespace tl {
namespace libfabric
{
    class controller;

    template<typename T>
    class unique_function : public std::function<T>
    {
        template<typename Fn, typename En = void>
        struct wrapper;

        // specialization for CopyConstructible Fn
        template<typename Fn>
        struct wrapper<Fn, std::enable_if_t< std::is_copy_constructible<Fn>::value >>
        {
            Fn fn;

            template<typename... Args>
            auto operator()(Args&&... args) { return fn(std::forward<Args>(args)...); }
        };

        // specialization for MoveConstructible-only Fn
        template<typename Fn>
        struct wrapper<Fn, std::enable_if_t< !std::is_copy_constructible<Fn>::value
            && std::is_move_constructible<Fn>::value >>
        {
            Fn fn;

            wrapper(Fn&& fn) : fn(std::forward<Fn>(fn)) { }

            wrapper(wrapper&&) = default;
            wrapper& operator=(wrapper&&) = default;

            // these two functions are instantiated by std::function
            // and are never called
            wrapper(const wrapper& rhs) : fn(const_cast<Fn&&>(rhs.fn)) { throw 0; } // hack to initialize fn for non-DefaultContructible types
            wrapper& operator=(wrapper&) { throw 0; }

            template<typename... Args>
            auto operator()(Args&&... args) { return fn(std::forward<Args>(args)...); }
        };

        using base = std::function<T>;

    public:
        unique_function() noexcept = default;
        unique_function(std::nullptr_t) noexcept : base(nullptr) { }

        template<typename Fn>
        unique_function(Fn&& f) : base(wrapper<Fn>{ std::forward<Fn>(f) }) { }

        unique_function(unique_function&&) = default;
        unique_function& operator=(unique_function&&) = default;

        unique_function& operator=(std::nullptr_t) { base::operator=(nullptr); return *this; }

        template<typename Fn>
        unique_function& operator=(Fn&& f)
        { base::operator=(wrapper<Fn>{ std::forward<Fn>(f) }); return *this; }

        using base::operator();
    };

}}}}

#endif
