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

#include <atomic>
//
#include <ghex/transport_layer/shared_message_buffer.hpp>
#include <ghex/transport_layer/libfabric/controller.hpp>
#include <ghex/transport_layer/libfabric/future.hpp>
//#include <ghex/transport_layer/libfabric/sender.hpp>

namespace gridtools {
    namespace ghex {

        // cppcheck-suppress ConfigurationNotChecked
        static hpx::debug::enable_print<false> com_deb("COMMUNI");
        static hpx::debug::enable_print<true> com_err("COMMUNI");

        namespace tl {
            namespace libfabric {
                namespace detail {

                    // -----------------------------------------------------------------
                    // an MPI error handling type that we can use to intercept
                    // MPI errors if we enable the error handler
                    static MPI_Errhandler hpx_mpi_errhandler = 0;

                    // extract MPI error message
                    std::string error_message(int code)
                    {
                        int N = 1023;
                        std::unique_ptr<char[]> err_buff(new char[std::size_t(N) + 1]);
                        err_buff[0] = '\0';

                        MPI_Error_string(code, err_buff.get(), &N);

                        return err_buff.get();
                    }

                    // function that converts an MPI error into an exception
                    void hpx_MPI_Handler(MPI_Comm*, int* errorcode, ...)
                    {
                        std::string temp = std::string("hpx_MPI_Handler : ") +
                                detail::error_message(*errorcode);
                        std::cout << temp << std::endl;
                        throw(std::runtime_error(temp));
                    }

                    // set an error handler for communicators that will be called
                    // on any error instead of the default behavior of program termination
                    void set_error_handler()
                    {
                        MPI_Comm_create_errhandler(
                            detail::hpx_MPI_Handler, &detail::hpx_mpi_errhandler);
                        MPI_Comm_set_errhandler(MPI_COMM_WORLD, detail::hpx_mpi_errhandler);
                    }
                }

                using region_provider    = libfabric_region_provider;
                using region_type        = rma::detail::memory_region_impl<region_provider>;

                struct status_t {};

                /** @brief common data which is shared by all communicators. This class is thread safe.
                  **/
                struct shared_communicator_state {
                    using rank_type = int;
                    using tag_type = std::uint64_t;
                    using controller_type = ghex::tl::libfabric::controller;

                    const mpi::rank_topology& m_rank_topology;
                    controller_type          *m_controller;
                    endpoint_wrapper         *m_rx_endpoint;
                    std::uintptr_t            m_ctag;
                    rank_type                 m_rank;
                    rank_type                 m_size;

                    shared_communicator_state(const mpi::rank_topology& t)
                        : m_rank_topology{t}
                        , m_rx_endpoint(nullptr)
                    {
                        const int random_msg_tag = 65535;
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(t.mpi_comm(), &m_rank));
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_size(t.mpi_comm(), &m_size));
                        if (m_rank==0) {
                            m_ctag = reinterpret_cast<std::uintptr_t>(this);
                            GHEX_DP_ONLY(com_deb, debug(hpx::debug::str<>("MPI send tag")
                                                        ,hpx::debug::hex<8>(m_ctag)));
                            for (int i=1; i<m_size; ++i) {
                                MPI_Send(&m_ctag, sizeof(std::uintptr_t), MPI_CHAR, i, random_msg_tag, t.mpi_comm());
                            }
                        }
                        else {
                            MPI_Status status;
                            MPI_Recv(&m_ctag, sizeof(std::uintptr_t), MPI_CHAR, 0, random_msg_tag, t.mpi_comm(), &status);
                            GHEX_DP_ONLY(com_deb, debug(hpx::debug::str<>("MPI recv tag")
                                                        ,hpx::debug::hex<8>(m_ctag)));
                        }
                    }

                    void init(controller_type *c)
                    {
                        m_controller  = c;
                        m_rx_endpoint = m_controller->get_rx_endpoint();
                    }
                };

                /** @brief communicator per-thread data.
                  **/
                struct communicator_state {
                    using shared_state_type      = shared_communicator_state;
                    using rank_type              = typename shared_state_type::rank_type;
                    using tag_type               = typename shared_state_type::tag_type;
                    template<typename T> using future = future_t<T>;
                    using progress_status        = ghex::tl::cb::progress_status;
                    using controller_type        = ghex::tl::libfabric::controller;

                    endpoint_wrapper *m_tx_endpoint = nullptr;
                    //
                    int  m_progressed_sends = 0;
                    int  m_progressed_recvs = 0;
                    int m_progressed_cancels = 0;
                    //

                    communicator_state() = default;
                    communicator_state(communicator_state &&) = default;

                    communicator_state(endpoint_wrapper *endpoint)
                    {
                        m_tx_endpoint = endpoint;
                    }

                    ~communicator_state()
                    {
                    }

                    void init(controller_type *c)
                    {
                        m_tx_endpoint = c->get_tx_endpoint();
                    }

                    progress_status progress() {
                        return {
                            std::exchange(m_progressed_sends,0),
                            std::exchange(m_progressed_recvs,0),
                            std::exchange(m_progressed_cancels,0)};
                    }
                };

                /** @brief completion handle returned from callback based communications
                  * @tparam ThreadPrimitives The thread primitives type */
                struct request_cb : public gridtools::ghex::tl::libfabric::request_t
                {
                    using shared_state_type = shared_communicator_state;
                    using state_type        = communicator_state;
                    using any_message_type  = ::gridtools::ghex::tl::libfabric::any_libfabric_message;
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

                /** @brief A communicator for point-to-point communication.
                  * This class is lightweight and copying/moving instances is safe and cheap.
                  * Communicators can be created through the context, and are thread-compatible.
                  */
                class communicator {
                  public: // member types
                    using shared_state_type      = shared_communicator_state;
                    using state_type             = communicator_state;
                    using rank_type              = int;
                    using tag_type               = typename shared_state_type::tag_type;
                    using request                = request_t;

                    template<typename T> using future = future_t<T>;
                    template<typename T> using allocator_type  = ghex::tl::libfabric::rma::memory_region_allocator<T>;
                    using lf_allocator_type  = allocator_type<unsigned char>;

                    using address_type              = rank_type;
                    using request_cb_type           = request_cb;
                    using message_type              = typename request_cb_type::any_message_type;
                    using any_message_type          = typename request_cb_type::any_message_type;
                    using libfabric_msg_type        = message_buffer<lf_allocator_type>;
                    using libfabric_sharedmsg_type  = shared_message_buffer<lf_allocator_type>;

                    using progress_status = ghex::tl::cb::progress_status;

                  private: // members
                    shared_state_type* m_shared_state;
                    state_type* m_state;

                  public: // ctors
                    communicator(shared_state_type* shared_state, state_type* state)
                    : m_shared_state{shared_state}
                    , m_state{state}
                    {
                        detail::set_error_handler();
                    }
                    communicator(const communicator&) = default;
                    communicator(communicator&&) = default;
                    communicator& operator=(const communicator&) = default;
                    communicator& operator=(communicator&&) = default;

                  public: // member functions
                    rank_type rank() const noexcept { return m_shared_state->m_rank; }
                    rank_type size() const noexcept { return m_shared_state->m_size; }
                    address_type address() const noexcept { return m_shared_state->m_rank; }

                    inline bool is_local(rank_type r) const noexcept { return m_shared_state->m_rank_topology.is_local(r); }
                    inline rank_type local_rank() const noexcept { return m_shared_state->m_rank_topology.local_rank(); }
                    inline auto mpi_comm() const noexcept { return m_shared_state->m_rank_topology.mpi_comm(); }

                    std::tuple<int,int,int,int, int, int> get_send_recv_counters() {
                        return std::make_tuple(
                                    int(m_shared_state->m_controller->sends_posted_),
                                    int(m_shared_state->m_controller->sends_complete),
                                    int(m_shared_state->m_controller->sends_readied_),
                                    int(m_shared_state->m_controller->recvs_posted_),
                                    int(m_shared_state->m_controller->recvs_complete),
                                    int(m_shared_state->m_controller->recvs_readied_));
                    }

                    // generate a tag with 0xaaaaaaRRRRtttttt address, rank, tag info
                    inline std::uint64_t make_tag64(std::uint32_t tag, std::uint32_t rank) {
                        return (
                                ((std::uint64_t(m_shared_state->m_ctag) & 0x0000000000FFFFFF) << 40) |
                                ((std::uint64_t(rank)                   & 0x000000000000FFFF) << 24) |
                                ((std::uint64_t(tag)                    & 0x0000000000FFFFFF))
                               );
                    }

                    inline rma::memory_pool<region_provider> *get_memory_pool() {
                        return &m_shared_state->m_controller->get_memory_pool_ptr();
                    }

                    // --------------------------------------------------------------------
                    // this takes a pinned memory region and sends it
                    void send_tagged_region(region_type *send_region, fi_addr_t dst_addr_, uint64_t tag_, void *ctxt)
                    {
                        [[maybe_unused]] auto scp = ghex::com_deb.scope(__func__);

                        // increment counter of total messages sent

                        GHEX_DP_ONLY(com_deb, debug(hpx::debug::str<>("send message buffer")
                                                    , "->", hpx::debug::dec<2>(dst_addr_)
                                                    , *send_region
                                                    , "tag", hpx::debug::hex<16>(tag_)));

                        bool ok = false;
                        int retries = 0, mult = 0;
                        while (!ok) {
                            ssize_t ret;
                            ret = fi_tsend(m_state->m_tx_endpoint->get_ep(),
                                           send_region->get_address(),
                                           send_region->get_message_length(),
                                           send_region->get_local_key(),
                                           dst_addr_, tag_, ctxt);
                            if (ret == 0) {
                                ++m_shared_state->m_controller->sends_posted_;
                                ok = true;
                            }
                            else if (ret == -FI_EAGAIN) {
                                com_deb.error("Reposting fi_sendv / fi_tsendv");
                                // no point stressing the system
                                progress();
                                //std::this_thread::sleep_for(std::chrono::microseconds(1));
//                                if (retries++>100) {
//                                    com_err.error("FI_EAGAIN fi_tsendv", hpx::debug::dec<5>(++mult));
//                                    retries = 0;
//                                }
                            }
                            else if (ret == -FI_ENOENT) {
                                // if a node has failed, we can recover
                                // @TODO : put something better here
                                com_deb.error("No destination endpoint, terminating.");
                                std::terminate();
                            }
                            else if (ret)
                            {
                                throw fabric_error(int(ret), "fi_sendv / fi_tsendv");
                            }
                        }
                    }

                    // --------------------------------------------------------------------
                    // the receiver posts a single receive buffer to the queue, attaching
                    // itself as the context, so that when a message is received
                    // the owning receiver is called to handle processing of the buffer
                    void receive_tagged_region(region_type *recv_region, fi_addr_t src_addr_, uint64_t tag_, void *ctxt)
                    {
                        [[maybe_unused]] auto scp = com_deb.scope(__func__);

                        GHEX_DP_ONLY(com_deb, debug(hpx::debug::str<>("recv message buffer")
                                                    , "<-", hpx::debug::dec<2>(src_addr_)
                                                    , *recv_region
                                                    , "tag", hpx::debug::hex<16>(tag_)));

                        // this should never actually return true and yield/sleep
                        bool ok = false;
                        int retries = 0, mult = 0;
                        while (!ok) {
                            uint64_t ignore = 0;
                            ssize_t ret = fi_trecv(m_shared_state->m_rx_endpoint->get_ep(),
                                recv_region->get_address(),
                                recv_region->get_size(),
                                recv_region->get_local_key(),
                                FI_ADDR_UNSPEC, tag_, ignore, ctxt);
                            if (ret ==0) {
                                ++m_shared_state->m_controller->recvs_posted_;
                                ok = true;
                            }
                            else if (ret == -FI_EAGAIN)
                            {
                                com_deb.error("reposting fi_trecv\n");
                                // no point stressing the system
                                progress();
                                //std::this_thread::sleep_for(std::chrono::microseconds(1));
//                                if (retries++>100) {
//                                    com_err.error("FI_EAGAIN fi_trecv", hpx::debug::dec<5>(++mult));
//                                    retries = 0;
//                                }
                            }
                            else if (ret != 0)
                            {
                                throw fabric_error(int(ret), "pp_post_rx");
                            }
                        }
                    }

                    /** @brief send a message. The message must be kept alive by the caller until the communication is
                     * finished.
                     * @tparam Message a message type
                     * @param msg an l-value reference to the message to be sent
                     * @param dst the destination rank
                     * @param tag the communication tag
                     * @return a future to test/wait for completion */
                    template<typename Message>
                    [[nodiscard]] future<void> send(const Message& msg, rank_type dst, tag_type tag)
                    {
                        [[maybe_unused]] auto scp = com_deb.scope(this, __func__, "(future)");

                        std::uint64_t stag = make_tag64(tag, this->rank());

                        // get main libfabric controller
                        auto controller = m_shared_state->m_controller;

                        // create a request
                        request req{
                            controller,
                            request_kind::send,
                            std::shared_ptr<context_info>(
                                        new context_info(m_state->m_tx_endpoint, stag, true))
                        };

                        req.m_lf_ctxt->user_cb_ = [this,p=req.m_lf_ctxt]() {
                            // cleanup temp region if necessary
                            p->message_holder_.clear();
                            p->message_region_ = nullptr;
                            GHEX_DP_ONLY(com_deb, debug(hpx::debug::str<>("Send (future)"), "F(set)"));
                            p->set_ready();
                            ++m_shared_state->m_controller->sends_readied_;
                        };
                        req.m_lf_ctxt->init_message_data(msg, stag);

                        GHEX_DP_ONLY(com_deb, debug(hpx::debug::str<>("Send (future)")
//                            , "thisrank", hpx::debug::dec<>(rank())
                            , "rank", hpx::debug::dec<>(dst)
                            , "tag", hpx::debug::hex<16>(std::uint64_t(tag))
                            , "ctag", hpx::debug::hex<8>(m_shared_state->m_ctag)
                            , "stag", hpx::debug::hex<16>(stag)
                            , "addr", hpx::debug::ptr(msg.data())
                            , "size", hpx::debug::hex<6>(msg.size())));

                        // async send, with callback to set the future ready when transfer is complete
                        send_tagged_region(req.m_lf_ctxt->message_region_, fi_addr_t(dst), stag, req.m_lf_ctxt.get());
                        // future constructor will be called with request as param
                        return req;
                    }

                    template<typename CallBack>
                    request_cb_type send(any_libfabric_message& msg, rank_type dst, tag_type tag, CallBack&& callback) {
                        GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type)
                        using V = typename std::remove_reference_t<any_libfabric_message>::value_type;
                        return send(message_type{libfabric_ref_message<V>{msg.data(),msg.size(), msg.m_holder.m_region}}, dst, tag, std::forward<CallBack>(callback));
                    }

                    /** @brief send a message and get notified with a callback when the communication has finished.
                      * The ownership of the message is transferred to this communicator and it is safe to destroy the
                      * message at the caller's site.
                      * Note, that the communicator has to be progressed explicitely in order to guarantee completion.
                      * @tparam CallBack a callback type with the signature void(any_message, rank_type, tag_type)
                      * @param msg r-value reference to any_message instance
                      * @param dst the destination rank
                      * @param tag the communication tag
                      * @param callback a callback instance
                      * @return a request to test (but not wait) for completion */
                    template<typename CallBack>
                    request_cb_type send(any_message_type&& msg, rank_type dst, tag_type tag, CallBack&& callback)
                    {
                        [[maybe_unused]] auto scp = com_deb.scope(this, __func__, "(callback)");

                        std::uint64_t stag = make_tag64(tag, this->rank());

                        // get main libfabric controller
                        auto controller = m_shared_state->m_controller;

                        // create a request
                        request req{
                            controller,
                            request_kind::send,
                            std::shared_ptr<context_info>(
                                        new context_info(m_state->m_tx_endpoint, stag, true))
                        };

                        req.m_lf_ctxt->init_message_data(msg, stag);

                        // now move message into callback
                        req.m_lf_ctxt->user_cb_ = [this,
                             p        = req.m_lf_ctxt,
                             msg      = std::move(msg),
                             callback = std::forward<CallBack>(callback),
                             dst, tag
                             ]() mutable
                        {
                            // cleanup temp region if necessary
                            p->message_holder_.clear();
                            p->message_region_ = nullptr;
                            p->set_ready();
                            ++m_shared_state->m_controller->sends_readied_;
                            GHEX_DP_ONLY(com_deb, debug(hpx::debug::str<>("Send any"),"(callback lambda)"
                                 , "F(set)", hpx::debug::dec<>(dst)));
                            callback(std::move(msg), dst, tag);
                            GHEX_DP_ONLY(com_deb, debug(hpx::debug::str<>("Send any"),"(callback lambda)"
                                 , "done", hpx::debug::dec<>(dst)));
                        };

                        GHEX_DP_ONLY(com_deb, debug(hpx::debug::str<>("Send any"), "(callback)"
//                         , "thisrank", hpx::debug::dec<>(rank())
                         , "rank", hpx::debug::dec<>(dst)
                         , "tag", hpx::debug::hex<16>(std::uint64_t(tag))
                         , "ctag", hpx::debug::hex<8>(m_shared_state->m_ctag)
                         , "stag", hpx::debug::hex<16>(stag)
                         , "addr", hpx::debug::ptr(msg.data())
                         , "size", hpx::debug::hex<6>(msg.size())));

                        // perform a send with the callback for when transfer is complete
                        send_tagged_region(req.m_lf_ctxt->message_region_, fi_addr_t(dst), stag, req.m_lf_ctxt.get());

                        return req;
                    }

                    /** @brief receive a message. The message must be kept alive by the caller until the communication is
                     * finished.
                     * @tparam Message a message type
                     * @param msg an l-value reference to the message to be sent
                     * @param src the source rank
                     * @param tag the communication tag
                     * @return a future to test/wait for completion */
                    template<typename Message>
                    [[nodiscard]] future<void> recv(Message &msg, rank_type src, tag_type tag)
                    {
                        [[maybe_unused]] auto scp = com_deb.scope(this, __func__, "(future)");
//                        GHEX_DP_ONLY(com_deb, debug(hpx::debug::str<>("map contents")
//                                            , m_shared_state->m_controller->get_memory_pool()->region_alloc_pointer_map_.debug_map()));

                        std::uint64_t stag = make_tag64(tag, src);

                        // main libfabric controller
                        auto controller = m_shared_state->m_controller;

                        // create a request
                        request req{
                            controller,
                            request_kind::recv,
                            std::shared_ptr<context_info>(
                                        new context_info(m_shared_state->m_rx_endpoint, stag, false))
                        };

                        req.m_lf_ctxt->init_message_data(msg, stag);

                        GHEX_DP_ONLY(com_deb, debug(hpx::debug::str<>("Recv (future)")
//                            , "thisrank", hpx::debug::dec<>(rank())
                            , "rank", hpx::debug::dec<>(src)
                            , "tag",  hpx::debug::hex<16>(std::uint64_t(tag))
                            , "ctag", hpx::debug::hex<8>(m_shared_state->m_ctag)
                            , "stag", hpx::debug::hex<16>(stag)
                            , "addr", hpx::debug::ptr(msg.data())
                            , "size", hpx::debug::hex<6>(msg.size())));

                        req.m_lf_ctxt->user_cb_ = [this,p=req.m_lf_ctxt](){
                            GHEX_DP_ONLY(com_deb, debug(hpx::debug::str<>("Recv (future)"), "F(set)"));
                            p->set_ready();
                            ++m_shared_state->m_controller->recvs_readied_;
                        };

                        receive_tagged_region(req.m_lf_ctxt->message_region_, fi_addr_t(src), stag, req.m_lf_ctxt.get());

                        return req;
                    }

                    template<typename CallBack>
                    request_cb_type recv(any_libfabric_message &&msg, rank_type src, tag_type tag, CallBack&& callback)
                    {
                        [[maybe_unused]] auto scp = com_deb.scope(this, __func__, "(callback)");
//                        GHEX_DP_ONLY(com_deb, debug(hpx::debug::str<>("map contents")
//                                            , m_shared_state->m_controller->memory_pool_->region_alloc_pointer_map_.debug_map()));

                        std::uint64_t stag = make_tag64(tag, src);

                        // main libfabric controller
                        auto controller = m_shared_state->m_controller;

                        // create a request
                        request req{
                            controller,
                            request_kind::recv,
                            std::shared_ptr<context_info>(
                                        new context_info(m_shared_state->m_rx_endpoint, stag, false))
                        };

                        req.m_lf_ctxt->init_message_data(msg, stag);

                        GHEX_DP_ONLY(com_deb, debug(hpx::debug::str<>("Recv (callback)")
//                            , "thisrank", hpx::debug::dec<>(rank())
                            , "rank", hpx::debug::dec<>(src)
                            , "tag", hpx::debug::hex<16>(std::uint64_t(tag))
                            , "ctag", hpx::debug::hex<8>(m_shared_state->m_ctag)
                            , "stag", hpx::debug::hex<16>(stag)
                            , "addr", hpx::debug::ptr(msg.data())
                            , "size", hpx::debug::hex<6>(msg.size())));

                        // now move message into callback
                        req.m_lf_ctxt->user_cb_ = [this,
                             p        = req.m_lf_ctxt,
                             msg      = std::move(msg), // std::forward<Msg>(msg)
                             callback = std::forward<CallBack>(callback),
                             src, tag
                             ]() mutable
                        {
                            GHEX_DP_ONLY(com_deb, debug(hpx::debug::str<>("Recv (callback)")
                                 , "F(set)", hpx::debug::dec<>(src)));
                            p->set_ready();
                            ++m_shared_state->m_controller->recvs_readied_;
                            callback(std::move(msg), src, tag);
                        };

                        // perform a send with the callback for when transfer is complete
                        receive_tagged_region(req.m_lf_ctxt->message_region_, fi_addr_t(src), stag, req.m_lf_ctxt.get());
                        return req;
                    }

//                    template<typename CallBack>
//                    request_cb_type recv(libfabric_sharedmsg_type &shared_msg, rank_type src, tag_type tag, CallBack&& callback)
//                    {
//                        return recv(shared_msg.get(), src, tag, std::forward<CallBack>(callback));
//                    }

                    /** @brief Function to poll the transport layer and check for completion of operations with an
                      * associated callback. When an operation completes, the corresponfing call-back is invoked
                      * with the message, rank and tag associated with this communication.
                      * @return non-zero if any communication was progressed, zero otherwise. */
                    progress_status progress() {
                        return m_shared_state->m_controller->poll_for_work_completions();
                    }
                };

            } // namespace libfabric
        } // namespace tl
    } // namespace ghex
} // namespace gridtools
