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
#ifndef INCLUDED_GHEX_TL_LIBFABRIC_COMMUNICATOR_CONTEXT_HPP
#define INCLUDED_GHEX_TL_LIBFABRIC_COMMUNICATOR_CONTEXT_HPP

#include <atomic>
//
#include <ghex/transport_layer/shared_message_buffer.hpp>
#include <ghex/transport_layer/libfabric/controller.hpp>
#include <ghex/transport_layer/libfabric/future.hpp>
#include <ghex/transport_layer/libfabric/sender.hpp>
#include <ghex/transport_layer/libfabric/rma/memory_pool.hpp>

namespace gridtools {
    namespace ghex {

        // cppcheck-suppress ConfigurationNotChecked
        static hpx::debug::enable_print<false> com_deb("COMMUNI");

        namespace tl {
            namespace libfabric {

                struct libfabric_tag_type {};

                struct status_t {};

                namespace impl
                {
                    template <typename T>
                    class by_value
                    {
                    private:
                        T _x;

                    public:
                        template <typename TFwd>
                        by_value(TFwd&& x) : _x{std::forward<TFwd>(x)}
                        {
                        }

                        auto&       get() &      { return _x; }
                        const auto& get() const& { return _x; }
                        auto        get() &&     { return std::move(_x); }
                    };

                    template <typename T>
                    class by_ref
                    {
                    private:
                        std::reference_wrapper<T> _x;

                    public:
                        by_ref(T& x) : _x{x}
                        {
                        }

                        auto&       get() &      { return _x.get(); }
                        const auto& get() const& { return _x.get(); }
                        auto        get() &&     { return std::move(_x.get()); }
                    };

                }

                // Unspecialized version: stores a `T` instance by value.
                template <typename T>
                struct fwd_capture_wrapper : impl::by_value<T>
                {
                    // "Inherit" constructors.
                    using impl::by_value<T>::by_value;
                };

                template <typename T>
                auto fwd_capture(T&& x)
                {
                    return fwd_capture_wrapper<T>(std::forward<T>(x));
                }

                // Specialized version: stores a `T` reference.
                template <typename T>
                struct fwd_capture_wrapper<T&> : impl::by_ref<T>
                {
                    // "Inherit" constructors.
                    using impl::by_ref<T>::by_ref;
                };

                /** @brief common data which is shared by all communicators. This class is thread safe.
                  * @tparam ThreadPrimitives The thread primitives type */
                template<typename ThreadPrimitives>
                struct shared_communicator_state {
                    using thread_primitives_type = ThreadPrimitives;
                    using transport_context_type = transport_context<libfabric_tag, ThreadPrimitives>;
                    using rank_type = int;
                    using tag_type = std::uint64_t;
                    using controller_type = ghex::tl::libfabric::controller;

                    controller_type        *m_controller;
                    transport_context_type *m_context;
                    thread_primitives_type *m_thread_primitives;

                    shared_communicator_state(controller_type *control, transport_context_type* tc, thread_primitives_type* tp)
                    : m_controller{control}
                    , m_context{tc}
                    , m_thread_primitives{tp}
                    {}

                    rank_type rank() const noexcept { return m_controller->m_rank_; }
                    rank_type size() const noexcept { return m_controller->m_size_; }
                };

                /** @brief communicator per-thread data.
                  * @tparam ThreadPrimitives The thread primitives type */
                template<typename ThreadPrimitives>
                struct communicator_state {
                    using shared_state_type = shared_communicator_state<ThreadPrimitives>;
                    using thread_primitives_type = ThreadPrimitives;
                    using thread_token = typename thread_primitives_type::token;
                    using rank_type = typename shared_state_type::rank_type;
                    using tag_type = typename shared_state_type::tag_type;
                    template<typename T>
                    using future = future_t<T>;
                    using progress_status = ghex::tl::cb::progress_status;
                    using controller_shared = std::shared_ptr<ghex::tl::libfabric::controller>;

                    thread_token* m_token_ptr;
                    int  m_progressed_sends = 0;
                    int  m_progressed_recvs = 0;
                    int m_progressed_cancels = 0;

                    communicator_state(thread_token* t)
                    : m_token_ptr{t}
                    {}

                    progress_status progress() {
                        return {
                            std::exchange(m_progressed_sends,0),
                            std::exchange(m_progressed_recvs,0),
                            std::exchange(m_progressed_cancels,0)};
                    }
                };

                /** @brief completion handle returned from callback based communications
                  * @tparam ThreadPrimitives The thread primitives type */
                template<typename ThreadPrimitives>
                struct request_cb : public gridtools::ghex::tl::libfabric::request_t
                {
                    using shared_state_type = shared_communicator_state<ThreadPrimitives>;
                    using state_type        = communicator_state<ThreadPrimitives>;
                    using message_type      = ::gridtools::ghex::tl::cb::any_message;
                    using tag_type          = typename shared_state_type::tag_type;
                    using rank_type         = int;
                    template <typename T>
                    using future            = future_t<T>;
                    using completion_type   = ::gridtools::ghex::tl::cb::request;
                    using queue_type = ::gridtools::ghex::tl::cb::callback_queue<future<void>, rank_type, tag_type>;

                    using gridtools::ghex::tl::libfabric::request_t::request_t;
                    using gridtools::ghex::tl::libfabric::request_t::m_controller;
                    using gridtools::ghex::tl::libfabric::request_t::m_kind;
                    using gridtools::ghex::tl::libfabric::request_t::m_lf_ctxt;

                    request_cb(const gridtools::ghex::tl::libfabric::request_t &r) {
                        m_controller = r.m_controller;
                        m_kind       = r.m_kind;
                        m_lf_ctxt    = r.m_lf_ctxt;
                    }
                };

                /** @brief A communicator for MPI point-to-point communication.
                  * This class is lightweight and copying/moving instances is safe and cheap.
                  * Communicators can be created through the context, and are thread-compatible.
                  * @tparam ThreadPrimitives The thread primitives type */
                template<typename ThreadPrimitives>
                class communicator {
                  public: // member types
                    using thread_primitives_type = ThreadPrimitives;
                    using shared_state_type = shared_communicator_state<ThreadPrimitives>;
                    using transport_context_type = transport_context<libfabric_tag, ThreadPrimitives>;
                    using thread_token = typename thread_primitives_type::token;
                    using state_type = communicator_state<ThreadPrimitives>;
                    using rank_type = int;
                    using tag_type = typename shared_state_type::tag_type;
                    using request = request_t;
                    using status = status_t;
                    template<typename T>
                    using future = future_t<T>;
                    template<typename T>
                    using allocator_type  = ghex::tl::libfabric::rma::allocator<T>;

                    using address_type    = rank_type;
                    using request_cb_type = request_cb<ThreadPrimitives>;
                    using message_type    = typename request_cb_type::message_type;
                    using progress_status = ghex::tl::cb::progress_status;

                  private: // members
                    shared_state_type* m_shared_state;
                    state_type* m_state;

                  public: // ctors
                    communicator(shared_state_type* shared_state, state_type* state)
                    : m_shared_state{shared_state}
                    , m_state{state}
                    {}
                    communicator(const communicator&) = default;
                    communicator(communicator&&) = default;
                    communicator& operator=(const communicator&) = default;
                    communicator& operator=(communicator&&) = default;

                  public: // member functions
                    rank_type rank() const noexcept { return m_shared_state->rank(); }
                    rank_type size() const noexcept { return m_shared_state->size(); }
                    address_type address() const noexcept { return rank(); }

                    inline std::uint64_t make_tag64(std::uint32_t tag) {
                        return (std::uint64_t(m_shared_state->m_context->ctag_) << 32) |
                                (std::uint64_t(tag) & 0xFFFFFFFF);
                    }

                    /** @brief send a message. The message must be kept alive by the caller until the communication is
                     * finished.
                     * @tparam Message a meassage type
                     * @param msg an l-value reference to the message to be sent
                     * @param dst the destination rank
                     * @param tag the communication tag
                     * @return a future to test/wait for completion */
                    template<typename Message>
                    [[nodiscard]] future<void> send(const Message& msg, rank_type dst, tag_type tag)
                    {
                        auto scp = ghex::com_deb.scope(this, __func__, "(future)");
                        ghex::com_deb.debug(hpx::debug::str<>("map"), m_shared_state->m_controller->memory_pool_->region_alloc_pointer_map_.debug_map());

                        std::uint64_t stag = make_tag64(tag);

                        // get main libfabric controller
                        auto controller = m_shared_state->m_controller;
                        // get a sender
                        ghex::tl::libfabric::sender *sndr = controller->get_sender(dst);

                        // create a request
                        std::shared_ptr<context_info> result(new context_info{false, nullptr});
                        request req{controller, request_kind::recv, std::move(result)};

                        ghex::com_deb.debug(hpx::debug::str<>("Send (future)")
                            , "thisrank", hpx::debug::dec<>(rank())
                            , "rank", hpx::debug::dec<>(dst)
                            , "tag", hpx::debug::hex<16>(std::uint64_t(tag))
                            , "cxt", hpx::debug::hex<8>(m_shared_state->m_context)
                            , "stag", hpx::debug::hex<16>(stag)
                            , "sndr", hpx::debug::ptr(sndr)
                            , "addr", hpx::debug::ptr(msg.data())
                            , "size", hpx::debug::hex<6>(msg.size()));

                        // async send, with callback to set the future ready when transfer is complete
                        sndr->message_region_external_ = true;
                        sndr->async_send_tagged(msg, stag, [p=req.m_lf_ctxt](){
                            p->m_ready = true;
                            ghex::com_deb.debug(hpx::debug::str<>("Send (future)"), "F(set)");
                        });

                        // future constructor will be called with request as param
                        return req;
                    }

                    /** @brief receive a message. The message must be kept alive by the caller until the communication is
                     * finished.
                     * @tparam Message a meassage type
                     * @param msg an l-value reference to the message to be sent
                     * @param src the source rank
                     * @param tag the communication tag
                     * @return a future to test/wait for completion */
                    template<typename Message>
                    [[nodiscard]] future<void> recv(Message& msg, rank_type src, tag_type tag)
                    {
                        auto scp = ghex::com_deb.scope(this, __func__, "(future)");
                        ghex::com_deb.debug(hpx::debug::str<>("map"), m_shared_state->m_controller->memory_pool_->region_alloc_pointer_map_.debug_map());

                        std::uint64_t stag = make_tag64(tag);

                        // main libfabric controller
                        auto controller = m_shared_state->m_controller;

                        // create a request
                        std::shared_ptr<context_info> result(new context_info{false, nullptr});
                        request req{controller, request_kind::recv, std::move(result)};

                        // get a receiver object (tagged, with a callback)
                        ghex::tl::libfabric::receiver *rcv =
                                controller->get_expected_receiver(src,
                            [p=req.m_lf_ctxt](){
                                p->m_ready = true;
                                ghex::com_deb.debug(hpx::debug::str<>("Receive (future)"), "F(set)");
                            });

                        // to support cancellation, we pass the context pointer (receiver)
                        // into the future shared state
                        req.m_lf_ctxt->m_lf_context = rcv;

                        ghex::com_deb.debug(hpx::debug::str<>("Recv (future)")
                            , "thisrank", hpx::debug::dec<>(rank())
                            , "rank", hpx::debug::dec<>(src)
                            , "tag", hpx::debug::hex<16>(std::uint64_t(tag))
                            , "cxt", hpx::debug::hex<8>(m_shared_state->m_context)
                            , "stag", hpx::debug::hex<16>(stag)
                            , "recv", hpx::debug::ptr(rcv)
                            , "addr", hpx::debug::ptr(msg.data())
                            , "size", hpx::debug::hex<6>(msg.size()));

                        rcv->pre_post_receive_msg(msg, stag);

                        return req;
                    }

                    /** @brief Function to poll the transport layer and check for completion of operations with an
                      * associated callback. When an operation completes, the corresponfing call-back is invoked
                      * with the message, rank and tag associated with this communication.
                      * @return non-zero if any communication was progressed, zero otherwise. */
                    progress_status progress() {
                        return m_shared_state->m_controller->poll_for_work_completions();
                    }

                   /** @brief send a message and get notified with a callback when the communication has finished.
                     * The ownership of the message is transferred to this communicator and it is safe to destroy the
                     * message at the caller's site.
                     * Note, that the communicator has to be progressed explicitely in order to guarantee completion.
                     * @tparam CallBack a callback type with the signature void(message_type, rank_type, tag_type)
                     * @param msg r-value reference to any_message instance
                     * @param dst the destination rank
                     * @param tag the communication tag
                     * @param callback a callback instance
                     * @return a request to test (but not wait) for completion */
                    template<typename CallBack>
                    request_cb_type send(message_type&& msg, rank_type dst, tag_type tag, CallBack&& callback)
                    {
                        auto scp = ghex::com_deb.scope(this, __func__, "(callback)");
                        ghex::com_deb.debug(hpx::debug::str<>("map"), m_shared_state->m_controller->memory_pool_->region_alloc_pointer_map_.debug_map());

                        std::uint64_t stag = make_tag64(tag);

                        // get main libfabric controller
                        auto controller = m_shared_state->m_controller;
                        // get a sender
                        ghex::tl::libfabric::sender *sndr = controller->get_sender(dst);
                        sndr->message_region_external_ = true;

                        // create a request
                        std::shared_ptr<context_info> result(new context_info{false, nullptr});
                        request req{controller, request_kind::recv, std::move(result)};

                        const void *data_ptr = msg.data();
                        std::size_t size = msg.size();

                        auto lambda = [
                                p=req.m_lf_ctxt,
                                callback = std::forward<CallBack>(callback),
                                msg = std::forward<message_type>(msg),
                                dst, tag
                                ]() mutable
                           {
                               p->m_ready = true;
                               ghex::com_deb.debug(hpx::debug::str<>("Send (callback)")
                                    , "F(set)", hpx::debug::dec<>(dst));
                               callback(std::move(msg), dst, tag);
                           };

                        ghex::com_deb.debug(hpx::debug::str<>("Send (callback)")
                            , "thisrank", hpx::debug::dec<>(rank())
                            , "rank", hpx::debug::dec<>(dst)
                            , "tag", hpx::debug::hex<16>(std::uint64_t(tag))
                            , "cxt", hpx::debug::hex<8>(m_shared_state->m_context)
                            , "stag", hpx::debug::hex<16>(stag)
                            , "sndr", hpx::debug::ptr(sndr)
                            , "addr", hpx::debug::ptr(msg.data())
                            , "size", hpx::debug::hex<6>(msg.size()));

                        // perform a send with the callback for when transfer is complete
                        sndr->async_send_tagged(data_ptr, size, stag, std::move(lambda));
                        return req;
                    }

                    template<typename CallBack>
                    request_cb_type recv(message_type&& msg, rank_type src, tag_type tag, CallBack&& callback)
                    {
                        auto scp = ghex::com_deb.scope(this, __func__, "(callback)");
                        ghex::com_deb.debug(hpx::debug::str<>("map"), m_shared_state->m_controller->memory_pool_->region_alloc_pointer_map_.debug_map());

                        std::uint64_t stag = make_tag64(tag);

                        // main libfabric controller
                        auto controller = m_shared_state->m_controller;

                        // create a request
                        std::shared_ptr<context_info> result(new context_info{false, nullptr});
                        request req{controller, request_kind::recv, std::move(result)};

                        // setup a callback that will set the future ready
                        // move the message into the callback function
                        auto lambda = [
                                p=req.m_lf_ctxt,
                                callback = std::forward<CallBack>(callback),
                                msg = std::move<message_type>(std::forward<message_type>(msg)),
                                src, tag
                                ]() mutable
                           {
                               p->m_ready = true;
                               ghex::com_deb.debug(hpx::debug::str<>("Recv (callback)")
                                    , "F(set)"
                                    , "from rank", hpx::debug::dec<>(src)
                                    , "triggering user callback");
                                // move the message into the user's final callback
                               callback(std::move(msg), src, tag);
                               ghex::com_deb.debug(hpx::debug::str<>("Recv (callback)")
                                    , "F(set)"
                                    , "from rank", hpx::debug::dec<>(src)
                                    , "done user callback");
                           };

                        // get a receiver object (tagged, with a callback)
                        ghex::tl::libfabric::receiver *rcv =
                                controller->get_expected_receiver(src, std::move(lambda));

                        // to support cancellation, we pass the context pointer (receiver)
                        // into the future shared state
                        req.m_lf_ctxt->m_lf_context = rcv;

                        ghex::com_deb.debug(hpx::debug::str<>("Recv (callback)")
                            , "thisrank", hpx::debug::dec<>(rank())
                            , "rank", hpx::debug::dec<>(src)
                            , "tag", hpx::debug::hex<16>(std::uint64_t(tag))
                            , "cxt", hpx::debug::hex<8>(m_shared_state->m_context)
                            , "stag", hpx::debug::hex<16>(stag)
                            , "recv", hpx::debug::ptr(rcv)
                            , "addr", hpx::debug::ptr(msg.data())
                            , "size", hpx::debug::hex<6>(msg.size()));

                        rcv->pre_post_receive_msg(msg, stag);
                        return req;
                    }

                    void barrier() {
                        if (auto token_ptr = m_state->m_token_ptr) {
                            auto& tp = *(m_shared_state->m_thread_primitives);
                            auto& token = *token_ptr;
                            tp.single(token, [this]() { MPI_Barrier(m_shared_state->m_controller->m_comm_); } );
                            progress(); // progress once more to set progress counters to zero
                            tp.barrier(token);
                        }
                        else
                            MPI_Barrier(m_shared_state->m_controller->m_comm_);
                    }
                };

            } // namespace libfabric
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_LIBFABRIC_COMMUNICATOR_CONTEXT_HPP */

