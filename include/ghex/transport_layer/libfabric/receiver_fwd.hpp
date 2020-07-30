#ifndef GHEX_LIBFABRIC_RECEIVER_FWD_HPP
#define GHEX_LIBFABRIC_RECEIVER_FWD_HPP

#include <ghex/transport_layer/libfabric/unique_function.hpp>
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

namespace gridtools {
namespace ghex {
namespace tl {
namespace libfabric
{
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
        friend class controller;

        typedef libfabric_region_provider                        region_provider;
        typedef rma::detail::memory_region_impl<region_provider> region_type;
        typedef boost::container::small_vector<region_type*,8>   zero_copy_vector;

        // internal handler type
        using postprocess_receiver_fn = unique_function<void(receiver*)>;

        // --------------------------------------------------------------------
        // constructor for a receiver object, creates a region and posts it
        receiver(fid_ep* endpoint,
                 rma::memory_pool<region_provider>& memory_pool,
                 postprocess_receiver_fn &&handler, bool prepost, int tag);

        // for vector storage, we need a move constructor
        receiver(receiver &&other) = default;

        // --------------------------------------------------------------------
        // destruct receive object
        ~receiver();

        // --------------------------------------------------------------------
        // The cleanup call deletes resources and sums counters from internals
        // once cleanup is done, the recevier should not be used, other than
        // dumping counters
        void cleanup();

        // --------------------------------------------------------------------
        // the receiver posts a single receive buffer to the queue, attaching
        // itself as the context, so that when a message is received
        // the owning receiver is called to handle processing of the buffer
        void pre_post_receive(uint64_t tag);
        void pre_post_receive_tagged(uint64_t tag);

        int cancel();

        template <typename Message>
        void pre_post_receive_msg(Message &msg, uint64_t tag);

        // --------------------------------------------------------------------
        // when a pre-posted receive completes, and rma transfers are needed
        // then an rma_receiver is used to coordinate
        rma_receiver* get_rma_receiver(fi_addr_t const& src_addr);

        // create an rma receiver to control fetching data, at startup we create
        // some and put them on a stack, then by default we just
        // take them from the stack when needed
        rma_receiver *create_rma_receiver(bool push_to_stack);

        // --------------------------------------------------------------------
        // A received message is routed by the controller into this function.
        // it might be an incoming message or just an ack sent to inform that
        // all rdma reads are complete from a previous send operation.
        int handle_recv(fi_addr_t const& src_addr, std::uint64_t len);
        int handle_recv_tagged(fi_addr_t const& src_addr, std::uint64_t len);

        // --------------------------------------------------------------------
        // A new connection only contains a locality address of the sender
        // - it can be handled directly without using an rma_receiver
        // - just get the address and add it to the address_vector
        //
        // This function is only called when lifabric is used for bootstrapping
        void handle_new_connection(controller *controller, std::uint64_t len);

        void handle_cancel();

        // --------------------------------------------------------------------
        // Not used, provided for potential/future rma_base compatibility
        void handle_error(struct fi_cq_err_entry err);

    private:
        fid_ep                            *endpoint_;
        region_type                       *header_region_;
        region_type                       *ghex_region_;
        rma::memory_pool<region_provider> *memory_pool_;

        // used by all receivers
        postprocess_receiver_fn            postprocess_handler_;

        // needed by tagged receivers
        fi_addr_t                         src_addr_;
        uint64_t                          tag_;
        unique_function<void(void)>       user_recv_cb_;

        // shared performance counters used by all receivers
        static performance_counter<unsigned int> messages_handled_;
        static performance_counter<unsigned int> acks_received_;
        static performance_counter<unsigned int> receives_pre_posted_;
        static performance_counter<unsigned int> active_rma_receivers_;

        // a stack of rma receiver objects
        typedef boost::lockfree::stack<
            rma_receiver*,
            boost::lockfree::capacity<GHEX_LIBFABRIC_MAX_EXPECTED>,
            boost::lockfree::fixed_sized<true>
        > rma_stack;
        static rma_stack rma_receivers_;
    };
}}}}

#endif
