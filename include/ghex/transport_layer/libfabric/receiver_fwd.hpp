#ifndef GHEX_LIBFABRIC_RECEIVER_FWD_HPP
#define GHEX_LIBFABRIC_RECEIVER_FWD_HPP

#include <ghex/transport_layer/libfabric/rma/detail/memory_region_impl.hpp>
#include <ghex/transport_layer/libfabric/rma/memory_pool.hpp>
#include <ghex/transport_layer/libfabric/rma/atomic_count.hpp>
//
#include <ghex/transport_layer/libfabric/performance_counter.hpp>
//
#include <ghex/transport_layer/libfabric/libfabric_region_provider.hpp>
#include <ghex/transport_layer/libfabric/header.hpp>
#include <ghex/transport_layer/libfabric/rma_base.hpp>
#include <ghex/transport_layer/libfabric/rma_receiver.hpp>
#include <ghex/transport_layer/libfabric/controller.hpp>
#include <ghex/transport_layer/libfabric/receiver_fwd.hpp>
//
#include <boost/container/small_vector.hpp>
//
#include <cstdint>

namespace ghex {
namespace tl {
namespace libfabric
{
    class controller;

    // The receiver is responsible for handling incoming messages. For that purpose,
    // it posts receive buffers. Incoming messages can be of two kinds:
    //      1) An ACK message which has been sent from an rma_receiver, to signal
    //         the sender about the successful retrieval of an incoming message.
    //      2) An incoming parcel, that consists of an header and an eventually
    //         piggy backed message. If the message is not piggy backed or zero
    //         copy RMA chunks need to be read, a rma_receiver is created to
    //         complete the transfer of the message
    struct receiver : public rma_base
    {
        typedef libfabric_region_provider                        region_provider;
        typedef rma::detail::memory_region_impl<region_provider> region_type;
        typedef boost::container::small_vector<region_type*,8>   zero_copy_vector;

        // --------------------------------------------------------------------
        // construct receive object
        receiver(fid_ep* endpoint,
                 rma::memory_pool<region_provider>& memory_pool);

        // --------------------------------------------------------------------
        // destruct receive object
        ~receiver();

        // --------------------------------------------------------------------
        // The cleanup call deletes resources and sums counters from internals
        // once cleanup is done, the recevier should not be used, other than
        // dumping counters
        void cleanup();

        // --------------------------------------------------------------------
        // Not used, provided for potential/future rma_base compatibility
        void handle_error(struct fi_cq_err_entry err);

        // --------------------------------------------------------------------
        // A new connection only contains a locality address of the sender
        // so it can be handled directly without creating an rma_receiver
        // just get the address and add it to the parclport address_vector
        bool handle_new_connection(controller *controller, std::uint64_t len);

        // --------------------------------------------------------------------
        // when a receive completes, this callback handler is called if
        // rma transfers are needed
        rma_receiver *create_rma_receiver(bool push_to_stack);

        // --------------------------------------------------------------------
        rma_receiver* get_rma_receiver(fi_addr_t const& src_addr);

        // --------------------------------------------------------------------
        // A received message is routed by the controller into this function.
        // it might be an incoming message or just an ack sent to inform that
        // all rdma reads are complete from a previous send operation.
        void handle_recv(fi_addr_t const& src_addr, std::uint64_t len);

        // --------------------------------------------------------------------
        // the receiver posts a single receive buffer to the queue, attaching
        // itself as the context, so that when a message is received
        // the owning receiver is called to handle processing of the buffer
        void pre_post_receive();

    private:
        fid_ep                            *endpoint_;
        region_type                       *header_region_ ;
        rma::memory_pool<region_provider> *memory_pool_;
        //
        friend class controller;

        // shared performance counters used by all receivers
        static performance_counter<unsigned int> messages_handled_;
        static performance_counter<unsigned int> acks_received_;
        static performance_counter<unsigned int> receives_pre_posted_;
        static performance_counter<unsigned int> active_rma_receivers_;

        //
        typedef boost::lockfree::stack<
            rma_receiver*,
            boost::lockfree::capacity<GHEX_LIBFABRIC_MAX_PREPOSTS>,
            boost::lockfree::fixed_sized<true>
        > rma_stack;
        static rma_stack rma_receivers_;
    };
}}}

#endif
