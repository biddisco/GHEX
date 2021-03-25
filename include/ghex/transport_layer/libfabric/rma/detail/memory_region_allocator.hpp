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

#include "ghex_libfabric_defines.hpp"
#include <ghex/transport_layer/libfabric/libfabric_region_provider.hpp>
#include <ghex/transport_layer/libfabric/rma/detail/memory_region_impl.hpp>
#include <ghex/transport_layer/libfabric/rma/memory_region.hpp>
#include <ghex/transport_layer/libfabric/rma/memory_pool.hpp>
//
#include <memory>

namespace gridtools {
namespace ghex {
namespace tl {
namespace libfabric {
namespace rma {

    // memory_region_pointer is a "fancy pointer" that can be dereferenced to
    // an address like any pointer, but it also stores memory region info
    // that contains pinned memory registration key for RMA operations
    template <typename T>
    struct memory_region_pointer
    {
        using mempool_type = ghex::tl::libfabric::rma::memory_pool<ghex::tl::libfabric::libfabric_region_provider, T>;
        using region_type = typename mempool_type::region_type;

        T *pointer_;
        region_type *region_;

        region_type *get_region() const {
            return region_;
        }

        // Constructors
        memory_region_pointer() noexcept
        : pointer_(nullptr)
        , region_(nullptr)
        { }

        memory_region_pointer(T * native, region_type* r) noexcept
        : pointer_(native)
        , region_(r)
        { }

        memory_region_pointer(memory_region_pointer const & rhs) noexcept
        : pointer_(rhs.pointer_)
        , region_(rhs.region_)
        { }

        memory_region_pointer(std::nullptr_t) noexcept
        {
            pointer_ = nullptr;
            region_  = nullptr;
        }

        template <typename U,
                  typename = typename std::enable_if<!std::is_same<T, U>::value && std::is_same<typename std::remove_cv<T>::type, U>::value>::type>
        memory_region_pointer(memory_region_pointer<U> const & rhs) noexcept
        : pointer_(rhs.pointer_)
        , region_(rhs.region_)
        { }

        template <typename U,
                  typename Dummy = void,
                  typename = typename std::enable_if<!std::is_same<typename std::remove_cv<T>::type, typename std::remove_cv<U>::type>::value && !std::is_void<U>::value,
                                                     decltype(static_cast<T *>(std::declval<U *>()))>::type>
        memory_region_pointer(memory_region_pointer<U> const & rhs) noexcept
        : pointer_(rhs.pointer_)
        , region_(rhs.region_)
        { }

        // NullablePointer requirements
        explicit operator bool() const noexcept
        {
            return pointer_ != nullptr;
        }

        memory_region_pointer & operator=(T *p) noexcept
        {
            pointer_ = p;
            region_  = nullptr;
            return *this;
        }

        memory_region_pointer & operator=(memory_region_pointer const & rhs) noexcept
        {
            if (this != &rhs)
            {
                pointer_ = rhs.pointer_;
                region_  = rhs.region_;
            }
            return *this;
        }

        memory_region_pointer & operator=(std::nullptr_t) noexcept
        {
            pointer_ = nullptr;
            region_  = nullptr;
            return *this;
        }

        // For pointer traits
        static memory_region_pointer pointer_to(T & x) {
            return memory_region_pointer(std::addressof(x));
        }

        // ---------------------------------------------
        // Random access iterator requirements (members)
        using iterator_category = std::random_access_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using reference = T &;
        using pointer = memory_region_pointer<T>;

        memory_region_pointer operator+(std::ptrdiff_t n) const
        {
            return memory_region_pointer(pointer_ + n, region_);
        }

        memory_region_pointer & operator+=(std::ptrdiff_t n)
        {
            pointer_ += n;
            return *this;
        }

        memory_region_pointer operator-(std::ptrdiff_t n) const
        {
            return memory_region_pointer(pointer_ - n, region_);
        }

        memory_region_pointer & operator-=(std::ptrdiff_t n)
        {
            pointer_ -= n;
            return *this;
        }

        std::ptrdiff_t operator-(memory_region_pointer const & rhs) const
        {
            return std::distance(rhs.pointer_, pointer_);
        }

        memory_region_pointer & operator++()
        {
            pointer_ += 1;
            return *this;
        }

        memory_region_pointer & operator--()
        {
            pointer_ -= 1;
            return *this;
        }

        memory_region_pointer operator++(int)
        {
            memory_region_pointer tmp(*this);
            ++*this;
            return tmp;
        }

        memory_region_pointer operator--(int)
        {
            memory_region_pointer tmp(*this);
            --*this;
            return tmp;
        }

        T * operator->() const noexcept { return pointer_; }
        T & operator*() const noexcept { return *pointer_; }
        T & operator[](std::size_t i) const noexcept { return pointer_[i]; }
    };

    // Random access iterator requirements (non-members)
    #define DEFINE_OPERATOR(oper, op, expr)                                                              \
      template <typename T>                                                                              \
      bool oper (memory_region_pointer<T> const & lhs, memory_region_pointer<T> const & rhs) noexcept                                  \
      { return expr; }                                                                                   \
      template <typename T>                                                                              \
      bool oper (memory_region_pointer<T> const & lhs, typename std::common_type<memory_region_pointer<T>>::type const & rhs) noexcept \
      { return lhs op rhs; }                                                                             \
      template <typename T>                                                                              \
      bool oper (typename std::common_type<memory_region_pointer<T>>::type const & lhs, memory_region_pointer<T> const & rhs) noexcept \
      { return lhs op rhs; }
    DEFINE_OPERATOR(operator==, ==, (lhs.pointer_ == rhs.pointer_))
    DEFINE_OPERATOR(operator!=, !=, (lhs.pointer_ != rhs.pointer_))
    DEFINE_OPERATOR(operator<,  <,  (lhs.pointer_ <  rhs.pointer_))
    DEFINE_OPERATOR(operator<=, <=, (lhs.pointer_ <= rhs.pointer_))
    DEFINE_OPERATOR(operator>,  >,  (lhs.pointer_ >  rhs.pointer_))
    DEFINE_OPERATOR(operator>=, >=, (lhs.pointer_ >= rhs.pointer_))
    #undef DEFINE_OPERATOR

    template <class T>
    struct memory_region_allocator {

        using value_type = T;

        // The commented out code represents functionality that
        // std::allocator_traits<allocator<T>> defaults for you,
        // if you do not provide it.
        // The implementation in the code shows you exactly what the defaults are.
        // Thus if you simply uncomment the code, you will not get any
        // functionality changes.
        // To get something different from what the defaults provide,
        // uncomment and change the implementation.
        // ------------------------------------
        using pointer = memory_region_pointer<value_type>;

        using const_pointer = typename std::pointer_traits<pointer>::template
                                                     rebind<value_type const>;
        using void_pointer       = typename std::pointer_traits<pointer>::template
                                                           rebind<void>;
        using const_void_pointer = typename std::pointer_traits<pointer>::template
                                                           rebind<const void>;

        using difference_type = typename std::pointer_traits<pointer>::difference_type;
        using size_type       = std::make_unsigned_t<difference_type>;

        template <class U> struct rebind {typedef memory_region_allocator<U> other;};
        // ------------------------------------

        // --------------------------------------------------
        // mempool and region types are specific to transport layer
        using mempool_type = ghex::tl::libfabric::rma::memory_pool<ghex::tl::libfabric::libfabric_region_provider, T>;
        using region_type = typename mempool_type::region_type;

        // all allocators share the same mempool
        static mempool_type *mempool_ptr;

        //        memory_region_allocator() = default;

        // --------------------------------------------------
        // default constructor
        memory_region_allocator() noexcept {}

        // --------------------------------------------------
        // default copy constructor
        template <typename U>
        memory_region_allocator(memory_region_allocator<U> const &) noexcept {}

        // --------------------------------------------------
        // default move constructor
        template<typename U>
        memory_region_allocator(memory_region_allocator<U>&&) noexcept {}

        void init_memory_pool(mempool_type *mempool) noexcept {
            mempool_ptr = mempool;
        }

        [[nodiscard]] pointer allocate(std::size_t n)
        {
            region_type *region = mempool_ptr->allocate_region(n);
            pointer p = pointer{reinterpret_cast<T*>(region->get_address()), region};
            //mempool_ptr->add_address_to_map(p.pointer_, region);
            return p;
        }

        void deallocate(pointer p, std::size_t)
        {
            // WARN: temporarily leaking memory due to MPICH bug
            //memory_region *region = mempool_ptr->region_from_address(ptr);
            //mempool_ptr->remove_address_from_map(p.pointer_, p.region_);
            mempool_ptr->deallocate(dynamic_cast<region_type*>(p.region_));
        }

        template <typename U>
        void construct(U* ptr) noexcept(std::is_nothrow_default_constructible<U>::value)
        {
            ::new(static_cast<void*>(ptr)) U;
        }

        template <typename U, typename...Args>
        void construct(U* ptr, Args&&... args)
        {
            ::new (static_cast<void*>(ptr)) U(std::forward<Args>(args)...);
        }
    };

    template <class T, class U>
    bool operator==(const memory_region_allocator<T>&, const memory_region_allocator<U>&) { return true; }
    template <class T, class U>
    bool operator!=(const memory_region_allocator<T>&, const memory_region_allocator<U>&) { return false; }

}}}}}

namespace gridtools {
namespace ghex {
namespace tl {
namespace libfabric {
namespace rma {

    template<>
    memory_region_allocator<unsigned char>::mempool_type* memory_region_allocator<unsigned char>::mempool_ptr = nullptr;

    template<>
    memory_region_allocator<double>::mempool_type* memory_region_allocator<double>::mempool_ptr = nullptr;

}}}}}
