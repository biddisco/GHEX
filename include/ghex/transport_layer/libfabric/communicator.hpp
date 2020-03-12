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

namespace ghex {
    // cppcheck-suppress ConfigurationNotChecked
    static hpx::debug::enable_print<true> com_deb("COMMUNI");

#undef FUNC_START_DEBUG_MSG
#undef FUNC_END_DEBUG_MSG
#define FUNC_START_DEBUG_MSG ::ghex::com_deb.debug(hpx::debug::str<>("*** Enter") , __func__);
#define FUNC_END_DEBUG_MSG   ::ghex::com_deb.debug(hpx::debug::str<>("### Exit ") , __func__);
}

namespace gridtools {
    namespace ghex {
        namespace tl {
            namespace libfabric {

                struct libfabric_tag_type {};

                struct status_t {};

                /** @brief common data which is shared by all communicators. This class is thread safe.
                  * @tparam ThreadPrimitives The thread primitives type */
                template<typename ThreadPrimitives>
                struct shared_communicator_state {
                    using thread_primitives_type = ThreadPrimitives;
                    using transport_context_type = transport_context<libfabric_tag, ThreadPrimitives>;
                    using rank_type = int;
                    using tag_type = int;
                    using controller_type = std::shared_ptr<::ghex::tl::libfabric::controller>;

                    controller_type m_controller;
                    transport_context_type* m_context;
                    thread_primitives_type* m_thread_primitives;

                    shared_communicator_state(controller_type control, transport_context_type* tc, thread_primitives_type* tp)
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
                    using queue_type = ::gridtools::ghex::tl::cb::callback_queue<future<void>, rank_type, tag_type>;
                    using progress_status = gridtools::ghex::tl::cb::progress_status;
                    using controller_type = std::shared_ptr<::ghex::tl::libfabric::controller>;

                    thread_token* m_token_ptr;
                    queue_type m_send_queue;
                    queue_type m_recv_queue;
                    int  m_progressed_sends = 0;
                    int  m_progressed_recvs = 0;

                    communicator_state(thread_token* t)
                    : m_token_ptr{t}
                    {}

                    progress_status progress() {
                        m_progressed_sends += m_send_queue.progress();
                        m_progressed_recvs += m_recv_queue.progress();
                        return {
                            std::exchange(m_progressed_sends,0),
                            std::exchange(m_progressed_recvs,0),
                            std::exchange(m_recv_queue.m_progressed_cancels,0)};
                    }
                };

                /** @brief completion handle returned from callback based communications
                  * @tparam ThreadPrimitives The thread primitives type */
                template<typename ThreadPrimitives>
                struct request_cb
                {
                    using shared_state_type = shared_communicator_state<ThreadPrimitives>;
                    using state_type        = communicator_state<ThreadPrimitives>;
                    using message_type      = ::gridtools::ghex::tl::cb::any_message;
                    using tag_type          = int;
                    using rank_type         = int;
                    template <typename T>
                    using future            = future_t<T>;
                    using completion_type   = ::gridtools::ghex::tl::cb::request;
                    using queue_type = ::gridtools::ghex::tl::cb::callback_queue<future<void>, rank_type, tag_type>;

                    queue_type* m_queue = nullptr;
                    completion_type m_completed;

                    bool test()
                    {
                        if(!m_queue) return true;
                        if (m_completed.is_ready())
                        {
                            m_queue = nullptr;
                            m_completed.reset();
                            return true;
                        }
                        return false;
                    }

                    bool cancel()
                    {
                        if(!m_queue) return false;
                        auto res = m_queue->cancel(m_completed.queue_index());
                        if (res)
                        {
                            m_queue = nullptr;
                            m_completed.reset();
                        }
                        return res;
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
                    using tag_type = int;
                    using request = request_t;
                    using status = status_t;
                    template<typename T>
                    using future = future_t<T>;
                    template<typename T>
                    using allocator_type  = ::ghex::tl::libfabric::rma::allocator<T>;

                    using address_type    = rank_type;
                    using request_cb_type = request_cb<ThreadPrimitives>;
                    using message_type    = typename request_cb_type::message_type;
                    using progress_status = gridtools::ghex::tl::cb::progress_status;

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

                    /** @brief send a message. The message must be kept alive by the caller until the communication is
                     * finished.
                     * @tparam Message a meassage type
                     * @param msg an l-value reference to the message to be sent
                     * @param dst the destination rank
                     * @param tag the communication tag
                     * @return a future to test/wait for completion */
                    template<typename Message>
                    [[nodiscard]] future<void> send(const Message& msg, rank_type dst, tag_type tag) {
                        request req;
                        FUNC_START_DEBUG_MSG
                        ::ghex::tl::libfabric::sender *sndr = m_shared_state->m_controller->get_sender(dst);

                        sndr->async_send(msg, tag);

/*
                        GHEX_CHECK_MPI_RESULT(MPI_Isend(reinterpret_cast<const void*>(msg.data()),
                                                        sizeof(typename Message::value_type) * msg.size(), MPI_BYTE,
                                                        dst, tag, m_shared_state->m_comm, &req.get()));
                        req.m_kind = request_kind::send;
*/
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
                    [[nodiscard]] future<void> recv(Message& msg, rank_type src, tag_type tag) {
                        request req;
                        FUNC_START_DEBUG_MSG

//                        m_shared_state->m_controller->
/*
                        GHEX_CHECK_MPI_RESULT(MPI_Irecv(reinterpret_cast<void*>(msg.data()),
                                                        sizeof(typename Message::value_type) * msg.size(), MPI_BYTE,
                                                        src, tag, m_shared_state->m_comm, &req.get()));
                        req.m_kind = request_kind::recv;
*/
                        return req;
                    }

                    /** @brief Function to poll the transport layer and check for completion of operations with an
                      * associated callback. When an operation completes, the corresponfing call-back is invoked
                      * with the message, rank and tag associated with this communication.
                      * @return non-zero if any communication was progressed, zero otherwise. */
                    progress_status progress() {
                        auto n = m_shared_state->m_controller->poll_for_work_completions();
//                        m_num_sends += other.m_num_sends;
//                        m_num_recvs += other.m_num_recvs;
//                        m_num_cancels += other.m_num_cancels;
                        progress_status temp{0,0,0};
                        return temp;
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
                        auto fut = send(msg, dst, tag);
                        if (fut.ready())
                        {
                            callback(std::move(msg), dst, tag);
                            ++(m_state->m_progressed_sends);
                            return {};
                        }
                        else
                        {
                            return { &m_state->m_send_queue,
                                m_state->m_send_queue.enqueue(std::move(msg), dst, tag, std::move(fut),
                                        std::forward<CallBack>(callback))};
                        }
                    }

                   /** @brief receive a message and get notified with a callback when the communication has finished.
                     * The ownership of the message is transferred to this communicator and it is safe to destroy the
                     * message at the caller's site.
                     * Note, that the communicator has to be progressed explicitely in order to guarantee completion.
                     * @tparam CallBack a callback type with the signature void(message_type, rank_type, tag_type)
                     * @param msg r-value reference to any_message instance
                     * @param src the source rank
                     * @param tag the communication tag
                     * @param callback a callback instance
                     * @return a request to test (but not wait) for completion */
                    template<typename CallBack>
                    request_cb_type recv(message_type&& msg, rank_type src, tag_type tag, CallBack&& callback)
                    {
                        auto fut = recv(msg, src, tag);
                        if (fut.ready())
                        {
                            callback(std::move(msg), src, tag);
                            ++(m_state->m_progressed_recvs);
                            return {};
                        }
                        else
                        {
                            return { &m_state->m_recv_queue,
                                m_state->m_recv_queue.enqueue(std::move(msg), src, tag, std::move(fut),
                                        std::forward<CallBack>(callback))};
                        }
                    }

                    void barrier() {
/*
                        if (auto token_ptr = m_state->m_token_ptr) {
                            auto& tp = *(m_shared_state->m_thread_primitives);
                            auto& token = *token_ptr;
                            tp.single(token, [this]() { MPI_Barrier(m_shared_state->m_comm); } );
                            progress(); // progress once more to set progress counters to zero
                            tp.barrier(token);
                        }
                        else
*/
                            MPI_Barrier(m_shared_state->m_controller->m_comm_);
                    }
                };

            } // namespace libfabric
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_LIBFABRIC_COMMUNICATOR_CONTEXT_HPP */

