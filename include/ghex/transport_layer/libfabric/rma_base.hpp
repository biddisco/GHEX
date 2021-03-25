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

#include <rdma/fi_eq.h>
#include <ghex/transport_layer/libfabric/libfabric_region_provider.hpp>
#include <ghex/transport_layer/libfabric/rma/memory_pool.hpp>
#include <ghex/transport_layer/callback_utils.hpp>

#ifdef SHOW_STACKTRACE
#define BOOST_STACKTRACE_USE_ADDR2LINE
#include <boost/stacktrace.hpp>
#endif

namespace gridtools { namespace ghex {
    // cppcheck-suppress ConfigurationNotChecked
    static hpx::debug::enable_print<false> any_deb("ANY_MSG");
}}

namespace gridtools {
namespace ghex {
namespace tl {
namespace libfabric
{
    enum rma_context_type {
        ctx_sender       = 0,
        ctx_receiver     = 1,
    };

    // rma_base is base class for sender, receiver and rma_receiver. The first
    // N bytes must be occupied by an fi_context structure for correct libfabric
    // operation. rma_base cannot therefore have any virtual methods.
    // The first bytes of this object storage must contain the fi_context
    // structure needed by libfabric.
    // we provide an extra 'type' enum so that we can dispatch calls to the correct
    // object type when an error occurs (see controller poll_send_queue etc)
    struct rma_base
    {
        // check address of context is address of this object
        rma_base(rma_context_type ctx_type)
            : context_rma_type(ctx_type)
        {
            HPX_ASSERT(reinterpret_cast<void*>(&this->context_reserved_space)
                        == reinterpret_cast<void*>(&*this));
        }

        inline const rma_context_type& context_type() const { return context_rma_type; }

    private:
        // libfabric requires some space for it's internal bookkeeping
        fi_context       context_reserved_space;
        rma_context_type context_rma_type;

    };

    struct libfabric_region_holder
    {
        using region_provider   = libfabric_region_provider;
        using region_type       = rma::detail::memory_region_impl<region_provider>;

        // needed for registering and unregistering memory
        static rma::memory_pool<libfabric_region_provider> *memory_pool_;

        // empty holder
        libfabric_region_holder() {
            m_region     = nullptr;
            m_unregister = false;
        }

        // reference an existing region, don't wipe on exit
        libfabric_region_holder(region_type *region) {
            m_region     = region;
            m_unregister = false;
        }

        // construct from pointer
        libfabric_region_holder(void *ptr, std::size_t size) {
            set_rma_from_pointer(ptr, size);
        }

        // move holder from one to another
        libfabric_region_holder(libfabric_region_holder && other)
        {
            m_region            = other.m_region;
            m_unregister        = other.m_unregister;
            // invalidate moved from container
            other.m_unregister  = false;
            other.m_region      = nullptr;
        }

        // move holder from one to another
        libfabric_region_holder & operator = (libfabric_region_holder && other)
        {
            m_region            = other.m_region;
            m_unregister        = other.m_unregister;
            // invalidate moved from container
            other.m_unregister  = false;
            other.m_region      = nullptr;
            return *this;
        }

        // on destruction, deregister the region if necessary
        ~libfabric_region_holder()
        {
            if (m_unregister) unregister();
        }

        void clear()
        {
            if (m_unregister && m_region) unregister();
            m_unregister = false;
            m_region     = nullptr;
        }

        void set_rma_region(region_type *region, std::size_t size)
        {
            m_unregister = false;
            m_region     = region;
            m_region->set_message_length(size);
        }

        // we only get rma from pointer as a 'last resort'
        // ideally, all buffers have regions associated with them
        void set_rma_from_pointer(const void *ptr, std::size_t size)
        {
#ifdef SHOW_STACKTRACE
            std::cout << "set_rma_from_pointer" << std::endl << boost::stacktrace::stacktrace() << std::endl;
#endif
            // did someone register this memory block and store it in the memory pool map
            m_region = dynamic_cast<region_type*>(
                        memory_pool_->region_from_address(ptr));
            // if the memory was not pinned, register it now
            // and mark the regions as needing to be unregistered on delete
            m_unregister = false;
            if (m_region == nullptr) {
                m_unregister = true;
                m_region = memory_pool_->register_temporary_region(ptr, size);
                memory_pool_->add_address_to_map(ptr, m_region);
                GHEX_DP_ONLY(any_deb, debug(hpx::debug::str<>("register"), *m_region));
            }
            m_region->set_message_length(size);
        }

        void unregister() {
            GHEX_DP_ONLY(any_deb, debug(hpx::debug::str<>("unregister"), *m_region));
            memory_pool_->remove_address_from_map(m_region->get_address(), m_region);
            memory_pool_->deallocate(m_region);
        }

        region_type *m_region;
        bool         m_unregister;
    };

    /** @brief simple wrapper around an l-value reference message (stores pointer and size)
      * @tparam T value type of the messages content */
    template<typename T>
    struct libfabric_ref_message
    {
        using region_provider   = libfabric_region_provider;
        using region_type       = rma::detail::memory_region_impl<region_provider>;

        using value_type = T;
        T* m_data;
        std::size_t m_size;
        region_type *m_region;
        T* data() noexcept { return m_data; }
        const T* data() const noexcept { return m_data; }
        std::size_t size() const noexcept { return m_size; }
    };

    /** @brief type erased message capable of holding any message. Uses optimized initialization for
      * ref_messages and std::shared_ptr pointing to messages. */
    struct any_libfabric_message
    {
        using value_type        = unsigned char;
        using region_provider   = libfabric_region_provider;
        using region_type       = rma::detail::memory_region_impl<region_provider>;
        template<typename T>
        using allocator_type    = rma::memory_region_allocator<T>;
        template<typename T>
        using libfabric_message_type = tl::message_buffer<allocator_type<T>>;

        unsigned char* __restrict               m_data;
        std::size_t                             m_size;
        std::unique_ptr<cb::any_message::iface> m_ptr;
        std::shared_ptr<char>                   m_ptr2;
        libfabric_region_holder                 m_holder;

        /** @brief Construct from an r-value: moves the message inside the type-erased structure.
          * Requires the message not to reallocate during the move. Note, that this operation will allocate
          * storage on the heap for the holder structure of the message.
          * @tparam Message a message type
          * @param m a message */
        template<class Message>
        explicit any_libfabric_message(Message&& m)
        : m_data{reinterpret_cast<unsigned char*>(m.data())}
        , m_size{m.size()*sizeof(typename Message::value_type)}
        , m_holder{}
        {
            // do this before moving the message
            init_rma(m);
            m_ptr = std::make_unique<cb::any_message::holder<Message>>(std::move(m));
        }


        /** @brief Construct from a reference: copies the pointer to the data and size of the data.
          * Note, that this operation will not allocate storage on the heap.
          * @tparam T a message type
          * @param m a ref_message to a message. */
        template<typename T>
        explicit any_libfabric_message(cb::ref_message<T>&& m)
        : m_data{reinterpret_cast<unsigned char*>(m.data())}
        , m_size{m.size()*sizeof(T)}
        , m_holder{}
        {
            init_rma(std::forward<cb::ref_message<T>>(m));
//            throw std::runtime_error("Help");
        }

        /** @brief Construct from a reference: copies the pointer to the data and size of the data.
          * Note, that this operation will not allocate storage on the heap.
          * @tparam T a message type
          * @param m a ref_message to a message. */
        template<typename T>
        explicit any_libfabric_message(libfabric_ref_message<T>&& m)
        : m_data{reinterpret_cast<unsigned char*>(m.data())}
        , m_size{m.size()*sizeof(T)}
        , m_holder{}
        {
            init_rma(m.m_region, m_size);
        }

        /** @brief Construct from a shared pointer: will share ownership with the shared pointer (aliasing)
          * and keeps the message wrapped by the shared pointer alive. Note, that this operation may
          * allocate on the heap, but does not allocate storage for the holder structure.
          * @tparam Message a message type
          * @param sm a shared pointer to a message*/
        template<typename Message>
        explicit any_libfabric_message(std::shared_ptr<Message>& sm)
        : m_data{reinterpret_cast<unsigned char*>(sm->data())}
        , m_size{sm->size()*sizeof(typename Message::value_type)}
        , m_ptr2(sm,reinterpret_cast<char*>(sm.get()))
        , m_holder{}
        {
            init_rma(*sm);
        }

        any_libfabric_message(any_libfabric_message&& other)
        {
            m_data   = other.m_data;
            m_size   = other.m_size;
            m_ptr    = std::move(other.m_ptr);
            m_ptr2   = std::move(other.m_ptr2);
            m_holder = std::move(other.m_holder);
        }

        any_libfabric_message& operator=(any_libfabric_message&&other)
        {
            m_data   = other.m_data;
            m_size   = other.m_size;
            m_ptr    = std::move(other.m_ptr);
            m_ptr2   = std::move(other.m_ptr2);
            m_holder = std::move(other.m_holder);
            return *this;
        }

        unsigned char* data() noexcept { return m_data;}
        const unsigned char* data() const noexcept { return m_data; }
        std::size_t size() const noexcept { return m_size; }

        region_type *region() const { return m_holder.m_region; }

        // #############################
        // setup RMA info
        // #############################
        void init_rma(void *ptr, std::size_t size)
        {
            m_holder.set_rma_from_pointer(ptr, size);
        }

        // reference message
        template<typename T>
        void init_rma(cb::ref_message<T>&& msg) {
            m_holder.set_rma_from_pointer(msg.data(), msg.size());
        }

        void init_rma(region_type *region, std::size_t size)
        {
            m_holder.set_rma_region(region, size);
        }

        // message buffer with a libfabric allocator
//        template <typename T>
        void init_rma(libfabric_message_type<unsigned char> & msg) {
            m_holder.set_rma_region(msg.get_buffer().m_pointer.region_, msg.size());
        }

        // shared pointer to message buffer with libfabric allocator
//        template <typename T>
        void init_rma(std::shared_ptr<libfabric_message_type<unsigned char>> &smsg) {
            m_holder.set_rma_region(
                        smsg->get_buffer().m_pointer.region_,
                        smsg->size());
        }


        // shared message buffer
//        template <typename T>
        void init_rma(tl::shared_message_buffer<allocator_type<unsigned char>> &smsg) {
            m_holder.set_rma_region(
                        smsg.m_message->get_buffer().m_pointer.region_,
                        smsg.m_message->size());
        }

        void init_rma(std::shared_ptr<tl::shared_message_buffer<allocator_type<unsigned char>>> &smsg) {
            m_holder.set_rma_region(
                        smsg->m_message->get_buffer().m_pointer.region_,
                        smsg->m_message->size());
        }

        // warn the user that using a std::allocator with libfabric
        // transport can lead to reduced performance
        void init_rma(tl::message_buffer<std::allocator<unsigned char>> &msg) {
            #pragma message(                                                         \
                "1 You are using a std::allocator with the libfabric transport layer." \
                "For improved efficiency, please use rma::memory_region_allocator<T>")
            m_holder.set_rma_from_pointer(msg.data(), msg.size());
        }

        // warn the user that using a std::allocator with libfabric
        // transport can lead to reduced performance
        void init_rma(tl::shared_message_buffer<std::allocator<unsigned char>> &msg) {
            #pragma message(                                                         \
                "2 You are using a std::allocator with the libfabric transport layer." \
                "For improved efficiency, please use rma::memory_region_allocator<T>")
            m_holder.set_rma_from_pointer(msg.data(), msg.size());
        }

        // warn the user that using a std::allocator with libfabric
        // transport can lead to reduced performance
        void init_rma(std::shared_ptr<tl::message_buffer<std::allocator<unsigned char>>> &smsg) {
            #pragma message(                                                         \
                "3 You are using a std::allocator with the libfabric transport layer." \
                "For improved efficiency, please use rma::memory_region_allocator<T>")
            m_holder.set_rma_from_pointer(smsg->data(), smsg->size());
        }

        // warn the user that using a std::allocator with libfabric
        // transport can lead to reduced performance
        void init_rma(std::shared_ptr<tl::shared_message_buffer<std::allocator<unsigned char>>> &smsg) {
            #pragma message(                                                         \
                "4 You are using a std::allocator with the libfabric transport layer." \
                "For improved efficiency, please use rma::memory_region_allocator<T>")
            m_holder.set_rma_from_pointer(
                        smsg->m_message->data(), smsg->m_message->size());
        }

        // catch all message type that covers std::vector etc
        template<typename Message>
        void init_rma(Message &msg) {
            m_holder.set_rma_from_pointer(msg.data(), msg.size());
        }


        //        template <typename Message>
        //        void init_rma(std::shared_ptr<Message> &smsg) {
        //            m_holder.set_rma_from_pointer(
        //                        smsg.m_message->get_buffer().m_pointer,
        //                        smsg.m_message->size());
        //        }



    };

}}}}

gridtools::ghex::tl::libfabric::rma::memory_pool<gridtools::ghex::tl::libfabric::libfabric_region_provider> *gridtools::ghex::tl::libfabric::libfabric_region_holder::memory_pool_ = nullptr;
