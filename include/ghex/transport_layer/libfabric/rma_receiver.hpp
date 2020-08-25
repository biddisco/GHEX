#ifndef GHEX_LIBFABRIC_RMA_RECEIVER_HPP
#define GHEX_LIBFABRIC_RMA_RECEIVER_HPP

#include <ghex/transport_layer/libfabric/unique_function.hpp>
#include <ghex/transport_layer/libfabric/rma/detail/memory_region_impl.hpp>
#include <ghex/transport_layer/libfabric/rma/memory_pool.hpp>
#include <ghex/transport_layer/libfabric/rma/atomic_count.hpp>
#include <ghex/transport_layer/libfabric/performance_counter.hpp>
//
#include <ghex/transport_layer/libfabric/libfabric_region_provider.hpp>
#include <ghex/transport_layer/libfabric/header.hpp>
#include <ghex/transport_layer/libfabric/rma_base.hpp>
#include <ghex/transport_layer/libfabric/libfabric_macros.hpp>
//
#include <boost/container/small_vector.hpp>
//
#include <vector>

namespace gridtools { namespace ghex {
    // cppcheck-suppress ConfigurationNotChecked
    static hpx::debug::enable_print<false> rma_deb("RMA_RCV");
#define DEB_PREFIX(c) hpx::debug::str<>(c), hpx::debug::ptr(this)
}}

namespace gridtools {
namespace ghex {
namespace tl {
namespace libfabric
{
    // The rma_receiver is responsible for receiving the
    // missing chunks of the message:
    //      1) Non-piggy backed non-zero copy chunks (if existing)
    //      2) The zero copy chunks from serialization
    struct rma_receiver : public rma_base
    {
        typedef libfabric_region_provider                        region_provider;
        typedef rma::detail::memory_region_impl<region_provider> region_type;
        typedef rma::memory_pool<region_provider>                memory_pool_type;
        typedef boost::container::small_vector<region_type*,8>   zero_copy_vector;

        typedef header<GHEX_LIBFABRIC_MESSAGE_HEADER_SIZE> header_type;
        static constexpr unsigned int header_size = header_type::header_block_size;

        typedef detail::chunktype chunktype;
        typedef std::function<void(rma_receiver*)> postprocess_handler;
        typedef std::function<void(void)>          user_handler;

        // --------------------------------------------------------------------
        rma_receiver(
            fid_ep* endpoint,
            memory_pool_type* memory_pool,
            postprocess_handler&& handler)
          : rma_base(ctx_rma_receiver)
          , endpoint_(endpoint)
          , header_region_(nullptr)
          , chunk_region_(nullptr)
          , message_region_(nullptr)
          , message_region_external_(false)
          , header_(nullptr)
          , memory_pool_(memory_pool)
          , rma_count_(0)
          , chunk_fetch_(false)
          , postprocess_handler_(std::move(handler))
        {}

        // --------------------------------------------------------------------
        // the main entry point when a message is received, this function
        // will dispatch to either read with or without rma depending on
        // whether there are zero copy chunks to handle
        int read_message(region_type* region, fi_addr_t const& src_addr)
        {
            [[maybe_unused]] auto scp = ghex::rma_deb.scope(this, __func__);
            ghex::rma_deb.debug(hpx::debug::str<>("map contents"), memory_pool_->region_alloc_pointer_map_.debug_map());
            HPX_ASSERT(rma_count_ == 0);
            HPX_ASSERT(header_ == nullptr);
            HPX_ASSERT(header_region_ == nullptr);
            HPX_ASSERT(chunk_region_ == nullptr);
//            HPX_ASSERT(message_region_ == nullptr);
            HPX_ASSERT(rma_regions_.size() == 0);
            HPX_ASSERT(chunk_fetch_ == false);

            // where this message came from
            src_addr_ = src_addr;

            // the region posted as a receive contains the received header
            header_region_ = region;
            header_        = reinterpret_cast<header_type*>(header_region_->get_address());

            HPX_ASSERT(header_);
            HPX_ASSERT(header_region_->get_address());

            rma_deb.debug(DEB_PREFIX("rma_receiver")
                , "read_message"
                , "Header :" , *header_);

//            rma_deb.trace(DEB_PREFIX("rma_receiver")
//                , "header memory"
//                , hpx::debug::mem_crc32(header_, header_->header_length(), "Header region (recv)"));


            // how many RMA operations are needed
            rma_count_ = header_->num_zero_copy_chunks();

            rma_deb.debug(DEB_PREFIX("rma_receiver")
                , "is expecting", hpx::debug::dec<>(rma_count_) , "read completions");

            handle_message_no_rma();
            ++msg_plain_;
            return 1;
        }

        // --------------------------------------------------------------------
        // Process a message that has no zero copy chunks
        void handle_message_no_rma()
        {
            [[maybe_unused]] auto scp = ghex::rma_deb.scope(this, __func__);
            ghex::rma_deb.debug(hpx::debug::str<>("map contents"), memory_pool_->region_alloc_pointer_map_.debug_map());
            HPX_ASSERT(header_);
            rma_deb.debug(DEB_PREFIX("rma_receiver")
                , "handle piggy backed send without zero copy regions");

            char *base_addr  = header_region_->get_address();
            char *piggy_back = header_->message_data();
            auto        size = header_->message_size();

            HPX_ASSERT(piggy_back);

//            rma_deb.trace(DEB_PREFIX("rma_receiver")
//                , hpx::debug::mem_crc32(piggy_back, size,
//                    "(Message region recv piggybacked - no rma - before memcpy)"));

            std::memcpy(base_addr, piggy_back, size);

//            rma_deb.trace(DEB_PREFIX("rma_receiver")
//                , hpx::debug::mem_crc32(piggy_back, size,
//                    "(Message region recv piggybacked - no rma - after memcpy)"));

            // message reading is complete
            rma_deb.debug(DEB_PREFIX("rma_receiver")
                , "message complete no RMA");
            if (user_recv_cb_) {
                user_recv_cb_();
            }

            cleanup_receive();
        }




        // --------------------------------------------------------------------
        // After message processing is complete, this routine cleans up and resets
        void cleanup_receive()
        {
            [[maybe_unused]] auto scp = ghex::rma_deb.scope(this, __func__);
            ghex::rma_deb.debug(hpx::debug::str<>("map contents"), memory_pool_->region_alloc_pointer_map_.debug_map());
            rma_deb.debug(DEB_PREFIX("rma_receiver")
                , "cleanup for receiver rma" , hpx::debug::ptr(this));
            //
            HPX_ASSERT(rma_count_ == 0);
            //
            ++recv_deletes_;
            //
            memory_pool_->deallocate(header_region_);
            header_region_ = nullptr;
            header_        = nullptr;
            src_addr_      = 0 ;
            user_recv_cb_  = [](){};
            //
            if (chunk_region_) {
                memory_pool_->deallocate(chunk_region_);
                chunk_region_  = nullptr;
            }
            //
            if (!message_region_external_ && message_region_) {
                memory_pool_->deallocate(message_region_);
                message_region_ = nullptr;
            }
            //
            for (auto region: rma_regions_) {
                memory_pool_->deallocate(region);
            }
            rma_regions_.clear();
            chunks_.clear();
            //
            rma_deb.debug(DEB_PREFIX("rma_receiver")
                , "Cleaned up, posting self back to rma stack");
            if (postprocess_handler_) postprocess_handler_(this);
        }

        // --------------------------------------------------------------------
        // called when the controller receives an error condition when
        // handling this object as an fi_context
        void handle_error(struct fi_cq_err_entry err)
        {
            rma_deb.error(DEB_PREFIX("rma_receiver")
                , "rma_receiver handling error"
                , hpx::debug::ptr(this));
            throw fabric_error(-int(err.err), "fi_read");
        }

    private:
        fid_ep                     *endpoint_;
        region_type                *header_region_;
        region_type                *chunk_region_;
        region_type                *message_region_;
        bool                        message_region_external_;
        header_type                *header_;
        std::vector<chunktype>      chunks_;
        zero_copy_vector            rma_regions_;
        memory_pool_type           *memory_pool_;
        fi_addr_t                   src_addr_;
        hpx::util::atomic_count     rma_count_;
        bool                        chunk_fetch_;
        postprocess_handler         postprocess_handler_;
        unique_function<void(void)> user_recv_cb_;

        double start_time_;

        friend struct receiver;
        friend class controller;

        // counters for statistics about messages
        static performance_counter<unsigned int> msg_plain_;
        static performance_counter<unsigned int> msg_rma_;
        static performance_counter<unsigned int> sent_ack_;
        static performance_counter<unsigned int> rma_reads_;
        static performance_counter<unsigned int> recv_deletes_;
    };

    performance_counter<unsigned int> rma_receiver::msg_plain_;
    performance_counter<unsigned int> rma_receiver::msg_rma_;
    performance_counter<unsigned int> rma_receiver::sent_ack_;
    performance_counter<unsigned int> rma_receiver::rma_reads_;
    performance_counter<unsigned int> rma_receiver::recv_deletes_;
}}}}

#endif
