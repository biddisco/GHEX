#ifndef GHEX_LIBFABRIC_RECEIVER_HPP
#define GHEX_LIBFABRIC_RECEIVER_HPP

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
//
#include <boost/container/small_vector.hpp>
//
#include <cstdint>

namespace ghex {
    // cppcheck-suppress ConfigurationNotChecked
    static hpx::debug::enable_print<true> recv_deb("RECEIVE");

#undef FUNC_START_DEBUG_MSG
#undef FUNC_END_DEBUG_MSG
#define FUNC_START_DEBUG_MSG ::ghex::recv_deb.debug(hpx::debug::str<>("*** Enter") , __func__);
#define FUNC_END_DEBUG_MSG   ::ghex::recv_deb.debug(hpx::debug::str<>("### Exit ") , __func__);
}

namespace ghex {
namespace tl {
namespace libfabric
{
        // --------------------------------------------------------------------
        // construct receive object
        receiver::receiver(fid_ep* endpoint,
                 rma::memory_pool<region_provider>& memory_pool)
            : rma_base(ctx_receiver)
            , endpoint_(endpoint)
            , header_region_(memory_pool.allocate_region(memory_pool.small_.chunk_size()))
            , memory_pool_(&memory_pool)
        {
            recv_deb.trace(hpx::debug::str<>("created receiver")
                , hpx::debug::ptr(this));
            // create an rma_receivers per receive and push it onto the rma stack
            create_rma_receiver(true);
            // Once constructed, we need to post the receive...
            pre_post_receive();
        }

        // --------------------------------------------------------------------
        // destruct receive object
        receiver::~receiver()
        {
            if (header_region_ && memory_pool_) {
                memory_pool_->deallocate(header_region_);
            }
            // this is safe to call twice - it might have been called already
            // to collect counter information by the fabric controller
            cleanup();
        }

        // --------------------------------------------------------------------
        // The cleanup call deletes resources and sums counters from internals
        // once cleanup is done, the receiver should not be used, other than
        // dumping counters
        void receiver::cleanup()
        {
            rma_receiver *rcv = nullptr;

            while (receiver::rma_receivers_.pop(rcv))
            {
//                msg_plain_    += rcv->msg_plain_;
//                msg_rma_      += rcv->msg_rma_;
//                sent_ack_     += rcv->sent_ack_;
//                rma_reads_    += rcv->rma_reads_;
//                recv_deletes_ += rcv->recv_deletes_;
                recv_deb.error("Cleanup" , "active", hpx::debug::dec<>(--active_rma_receivers_));
                delete rcv;
            }
        }

        // --------------------------------------------------------------------
        // Not used, provided for potential/future rma_base compatibility
        void receiver::handle_error(struct fi_cq_err_entry err) {}

        // --------------------------------------------------------------------
        // A new connection only contains a locality address of the sender
        // so it can be handled directly without creating an rma_receiver
        // just get the address and add it to the parclport address_vector
        bool receiver::handle_new_connection(controller *controller, std::uint64_t len)
        {
            FUNC_START_DEBUG_MSG;
            recv_deb.debug(hpx::debug::str<>("new connection")
                , "length ", hpx::debug::dec<>(len)
                , "pre-posted " , hpx::debug::dec<>(--receives_pre_posted_));

            // We save the received region and swap it with a newly allocated one
            // so that we can post a recv again as soon as possible.
            region_type* region = header_region_;
            header_region_ = memory_pool_->allocate_region(memory_pool_->small_.chunk_size());
            pre_post_receive();


            recv_deb.trace(hpx::debug::str<>("header")
                ,  hpx::debug::mem_crc32(region->get_address()
                ,len, "Header region (new connection)"));

            rma_receiver::header_type *header =
                    reinterpret_cast<rma_receiver::header_type*>(region->get_address());

            // The message size should match the locality data size
            HPX_ASSERT(header->message_size() == locality_defs::array_size);

            libfabric::locality source_addr;
            std::memcpy(source_addr.fabric_data_writable(),
                        header->message_data(),
                        locality_defs::array_size);
            recv_deb.debug(hpx::debug::str<>("bootstrap")
                , "Received connection locality"
                , iplocality(source_addr));

            // free up the region we consumed
            memory_pool_->deallocate(region);

            // Add the sender's address to the address vector and update it
            // with the fi_addr address vector table index (rank)
            source_addr = controller->insert_address(source_addr);
            controller->update_bootstrap_connections();

            FUNC_END_DEBUG_MSG;
            return true;
        }

        // --------------------------------------------------------------------
        // when a receive completes, this callback handler is called if
        // rma transfers are needed
        rma_receiver *receiver::create_rma_receiver(bool push_to_stack)
        {
            // this is the rma_receiver completion handling function
            // it just returns the rma_receiver back to the stack
            auto f = [](rma_receiver* recv)
            {
                ++active_rma_receivers_;
                recv_deb.debug(hpx::debug::str<>("rma_receiver")
                    , "Push"
                    , "active", hpx::debug::dec<>(active_rma_receivers_));
                if (!receiver::rma_receivers_.push(recv)) {
                    // if the capacity overflowed, just delete this one
                    --active_rma_receivers_;
                    recv_deb.debug(hpx::debug::str<>("stack full 1")
                        , "active", hpx::debug::dec<>(active_rma_receivers_));
                    delete recv;
                }
            };

            // Put a new rma_receiver on the stack
            rma_receiver *recv = new rma_receiver(/*parcelport_, */endpoint_, memory_pool_, std::move(f));
            ++active_rma_receivers_;
            recv_deb.debug(hpx::debug::str<>("rma_receiver")
                , "Create new"
                , hpx::debug::dec<>(active_rma_receivers_));
            if (push_to_stack) {
                if (!receiver::rma_receivers_.push(recv)) {
                    // if the capacity overflowed, just delete this one
                    --active_rma_receivers_;
                    recv_deb.debug(hpx::debug::str<>("stack full 2")
                        , hpx::debug::dec<>(active_rma_receivers_));
                    delete recv;
                }
            }
            else {
                return recv;
            }
            return nullptr;
        }

        // --------------------------------------------------------------------
        rma_receiver* receiver::get_rma_receiver(fi_addr_t const& src_addr)
        {
            rma_receiver *recv = nullptr;
            // cannot yield here - might be called from background thread
            if (!receiver::rma_receivers_.pop(recv)) {
                recv = create_rma_receiver(false);
            }
            --active_rma_receivers_;
            recv_deb.debug(hpx::debug::str<>("get_rma_receiver")
                , "active", hpx::debug::dec<>(active_rma_receivers_));
            //
            recv->src_addr_       = src_addr;
            recv->endpoint_       = endpoint_;
            recv->header_region_  = nullptr;
            recv->chunk_region_   = nullptr;
            recv->message_region_ = nullptr;
            recv->header_         = nullptr;
            recv->rma_count_      = 0;
            recv->chunk_fetch_    = false;
            return recv;
        }


        // --------------------------------------------------------------------
        // A received message is routed by the controller into this function.
        // it might be an incoming message or just an ack sent to inform that
        // all rdma reads are complete from a previous send operation.
        void receiver::handle_recv(fi_addr_t const& src_addr, std::uint64_t len)
        {
            FUNC_START_DEBUG_MSG;
            static_assert(sizeof(std::uint64_t) == sizeof(std::size_t),
                "sizeof(std::uint64_t) != sizeof(std::size_t)");

            recv_deb.debug(hpx::debug::str<>("handling recv")
                , "pre-posted" , hpx::debug::dec<>(--receives_pre_posted_));

            // If we receive a message of 8 bytes, we got a tag and need to handle
            // the tag completion...
            if (len <= sizeof(std::uint64_t))
            {
                // @TODO: fixme immediate tag retrieval
                // Get the sender that has completed rma operations and signal to it
                // that it can now cleanup - all remote get operations are done.
                sender* snd = *reinterpret_cast<sender **>(header_region_->get_address());
                pre_post_receive();
                recv_deb.debug(hpx::debug::str<>("RMA ack")
                    , hpx::debug::ptr(snd));
                ++acks_received_;
                snd->handle_message_completion_ack();
                return;
            }

            rma_receiver* recv = get_rma_receiver(src_addr);

            // We save the received region and swap it with a newly allocated one
            // so that we can post a recv again as soon as possible.
            region_type* region = header_region_;
            header_region_ = memory_pool_->allocate_region(memory_pool_->small_.chunk_size());
            pre_post_receive();

            // we dispatch our work to our rma_receiver once it completed the
            // prior message. The saved region is passed to the rma handler
            ++messages_handled_;
            recv->read_message(region, src_addr);

            FUNC_END_DEBUG_MSG;
        }

        // --------------------------------------------------------------------
        // the receiver posts a single receive buffer to the queue, attaching
        // itself as the context, so that when a message is received
        // the owning receiver is called to handle processing of the buffer
        void receiver::pre_post_receive()
        {
            FUNC_START_DEBUG_MSG;
            void *desc = header_region_->get_local_key();
            recv_deb.debug(hpx::debug::str<>("Pre-Posting")
                , *header_region_
                , "context " , hpx::debug::ptr(this)
                , "pre-posted " , hpx::debug::dec<>(++receives_pre_posted_));

            // this should never actually return true and yield
            bool ok = false;
            while(!ok) {
                // post a receive using 'this' as the context, so that this
                // receiver object can be used to handle the incoming
                // receive/request
                ssize_t ret = fi_recv(this->endpoint_,
                    this->header_region_->get_address(),
                    this->header_region_->get_size(), desc, 0, this);

                if (ret ==0) {
                    ok = true;
                }
                else if (ret == -FI_EAGAIN)
                {
                    recv_deb.error("reposting fi_recv\n");
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
                else if (ret != 0)
                {
                    throw fabric_error(int(ret), "pp_post_rx");
                }
            }
            FUNC_END_DEBUG_MSG;
        }

        performance_counter<unsigned int> receiver::messages_handled_(0);
        performance_counter<unsigned int> receiver::acks_received_(0);
        performance_counter<unsigned int> receiver::receives_pre_posted_(0);
        performance_counter<unsigned int> receiver::active_rma_receivers_(0);
        receiver::rma_stack               receiver::rma_receivers_;

}}}

#endif
