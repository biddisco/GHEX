#ifndef GHEX_LIBFABRIC_RMA_RECEIVER_HPP
#define GHEX_LIBFABRIC_RMA_RECEIVER_HPP

#include <ghex/transport_layer/libfabric/unique_function.hpp>
#include <ghex/transport_layer/libfabric/rma/detail/memory_region_impl.hpp>
#include <ghex/transport_layer/libfabric/rma/memory_pool.hpp>
#include <ghex/transport_layer/libfabric/rma/detail/memory_region_allocator.hpp>
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

namespace ghex {
    // cppcheck-suppress ConfigurationNotChecked
    static hpx::debug::enable_print<true> rma_deb("RMA_RCV");
#define DEB_PREFIX(c) hpx::debug::str<>(c), hpx::debug::ptr(this)
}

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
            ghex::rma_deb.scope(this, __func__);
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

            if (header_->bootstrap()) {
//                parcelport_->set_bootstrap_complete();
                return handle_bootstrap_message();
            }

            if (header_->chunk_ptr()==nullptr) {
                // the header does not have piggybacked chunks, we must rma-get them before
                // we can decode the message, they may need further rma-get operations
                handle_message_no_chunk_data();
                return 0;
            }

            // how many RMA operations are needed
            rma_count_ = header_->num_zero_copy_chunks();

            rma_deb.debug(DEB_PREFIX("rma_receiver")
                , "is expecting", hpx::debug::dec<>(rma_count_) , "read completions");

            // If we have no zero copy chunks and piggy backed data, we can
            // process the message immediately, otherwise, dispatch to receiver
            // If we have neither piggy back, nor zero copy chunks, rma_count is 0
            if (rma_count_ == 0)
            {
                handle_message_no_rma();
                ++msg_plain_;
                return 1;
            }
            else {
                handle_message_with_zerocopy_rma();
                ++msg_rma_;
                return 0;
            }
        }

        // --------------------------------------------------------------------
        // @TODO insert docs here
        int handle_bootstrap_message()
        {
            ghex::rma_deb.scope(this, __func__);
            rma_deb.debug(DEB_PREFIX("rma_receiver"), "handle bootstrap");
            HPX_ASSERT(header_);

            char *piggy_back = header_->message_data();
            HPX_ASSERT(piggy_back);

//            rma_deb.trace(DEB_PREFIX("rma_receiver")
//                , hpx::debug::mem_crc32(piggy_back, header_->message_size(),
//                    "(Message/bootstrap region recv piggybacked - no rma)"));
            //
            std::size_t N = header_->message_size()/sizeof(libfabric::locality);
            //
            std::vector<libfabric::locality> addresses;
            addresses.reserve(N);
            //
            const libfabric::locality *data =
                    reinterpret_cast<libfabric::locality*>(header_->message_data());
            for (std::size_t i=0; i<N; ++i) {
                addresses.push_back(data[i]);
                rma_deb.debug(DEB_PREFIX("rma_receiver")
                    , hpx::debug::str<>("bootstrap received"), iplocality(data[i]));
            }
            rma_deb.debug(hpx::debug::str<>("bootstrap received"), hpx::debug::dec<>(N) , "addresses");
            // controller_->recv_bootstrap_address(addresses);
            //
            cleanup_receive();            
            return 1;
        }

        // --------------------------------------------------------------------
        // Process a message that has no zero copy chunks
        void handle_message_no_rma()
        {
            ghex::rma_deb.scope(this, __func__);
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
        // Process a message that has zero copy chunks. for each chunk we
        // make an RMA read request
        void handle_message_with_zerocopy_rma()
        {
            ghex::rma_deb.scope(this, __func__);
            chunks_.resize(header_->num_chunks());
            char *chunk_data = header_->chunk_data();
            HPX_ASSERT(chunk_data);

            size_t chunkbytes =
                chunks_.size() * sizeof(chunktype);

            std::memcpy(chunks_.data(), chunk_data, chunkbytes);
            rma_deb.debug(DEB_PREFIX("rma_receiver")
                , "Copied chunk data from header : size"
                , hpx::debug::dec<>(chunkbytes));

            if (rma_deb.is_enabled()) {
                for (const chunktype &c : chunks_)
                {
                    rma_deb.debug(DEB_PREFIX("rma_receiver")
                        , "recv : chunk : size" , hpx::debug::hex<6>(c.size_)
                        , "type"   , hpx::debug::dec<>((uint64_t)c.type_)
                        , "rkey"   , hpx::debug::ptr(c.rma_)
                        , "cpos"   , hpx::debug::ptr(c.data_.cpos_)
                        , "index"  , hpx::debug::dec<>(c.data_.index_));
                }
            }
            rma_regions_.reserve(rma_count_);

            // for each zerocopy chunk, schedule a read operation
            read_chunk_list();
        }

        // --------------------------------------------------------------------
        // Process a message where the chunk information did not fit into
        // the header. An extra RMA read of chunk data must be made before
        // the chunks can be identified (and possibly retrieved from the remote node)
        void handle_message_no_chunk_data()
        {
            ghex::rma_deb.scope(this, __func__);
            throw fabric_error(0, "GHEX unsupported handle_message_no_chunk_data");
        }

        // --------------------------------------------------------------------
        // After remote chunks have been read by rma, process the chunk list
        // and initiate further rma reads if necessary
        int handle_chunks_read_message()
        {
            ghex::rma_deb.scope(this, __func__);
            char *chunk_data = chunk_region_->get_address();
            HPX_ASSERT(chunk_data);
            //
            uint64_t chunkbytes = chunk_region_->get_message_length();
            uint64_t num_chunks = chunkbytes/sizeof(chunktype);
            chunks_.resize(num_chunks);
            std::memcpy(chunks_.data(), chunk_data, chunkbytes);
            rma_deb.debug(DEB_PREFIX("rma_receiver")
                , "Copied chunk data from chunk_region: size" , hpx::debug::dec<>(chunkbytes)
                , "with num chunks" , hpx::debug::dec<>(num_chunks));
            //
            HPX_ASSERT(rma_regions_.size() == 0);
            //
            chunk_fetch_ = false;
            // for each zerocopy chunk, schedule a read operation
            uint64_t zc_count =
                std::count_if(chunks_.begin(), chunks_.end(), [](chunktype &c) {
                    return c.type_ == detail::chunk_type_pointer ||
                           c.type_ == detail::chunk_type_rma;
                });
            // this is the number of rma-completions we must wait for
            rma_count_ = zc_count;
            //
            rma_deb.debug(DEB_PREFIX("rma_receiver")
                , "Restarting RMA reads with" , hpx::debug::dec<>(zc_count) , "rma chunks");
            // do not return rma_count_ as it might already have decremented! (racey)
            read_chunk_list();
            return zc_count;
        }

        // --------------------------------------------------------------------
        // Each RMA read completion will enter this function and count down until
        // all are done, then we can process the message and cleanup
        //
        // whenever an rma read completion event occurs, this function is entered.
        // an atomic counter, counts down with each completion, until we hit zero,
        // when all expected read results are available.
        //
        // Return true if an ack was sent during this handler and the
        // user receive callback was triggered
        bool handle_rma_read_completion()
        {
            ghex::rma_deb.scope(this, __func__);;
            HPX_ASSERT(rma_count_ > 0);
            // If we haven't read all chunks, we can return and wait
            // for the other incoming read completions
            if (--rma_count_ > 0)
            {
                rma_deb.debug(DEB_PREFIX("rma_receiver")
                    , "Not yet read all RMA regions" , hpx::debug::ptr(this));
                ghex::rma_deb.scope(this, __func__);;
                return false;
            }

            HPX_ASSERT(rma_count_ == 0);

            // when the chunk structure could not be piggybacked, the chunk_fetch_
            // flag is set prior to reading the chunks.
            if (chunk_fetch_) {
                rma_deb.debug(DEB_PREFIX("rma_receiver")
                    , "rma read chunk list complete");
                if (handle_chunks_read_message()>0) {
                    // more rma reads have been started, so exit and wait for them
                    return false;
                }
            }
            else {
                rma_deb.debug(DEB_PREFIX("rma_receiver")
                    , "all RMA regions now read");
            }

            // If the main message was not piggy backed, then the message region
            // is either the final chunk of the rma list (if chunks were piggybacked)
            // or read via rma during the chunk fetch (chunks not piggybacked)
            if (!header_->message_piggy_back())
            {
                if (header_->chunk_ptr()) {
                    message_region_ = rma_regions_.back();
                    //
                    rma_regions_.resize(rma_regions_.size()-1);
                    chunks_.resize(chunks_.size()-1);
                }
                else {
                    rma_deb.debug(DEB_PREFIX("rma_receiver")
                        , "No piggy back message or chunks");
                    // message region should have been read by handle_message_no_chunk_data
                    HPX_ASSERT(message_region_);
                }
            }

            std::size_t message_length = header_->message_size();
            char *message = nullptr;
            if (message_region_)
            {
                message = static_cast<char *>(message_region_->get_address());
                HPX_ASSERT(message);
                rma_deb.debug(DEB_PREFIX("rma_receiver")
                    , "No piggy_back RMA message"
                    , "region" , hpx::debug::ptr(message_region_)
                    , "address" , hpx::debug::ptr(message_region_->get_address())
                    , "length" , hpx::debug::hex<6>(message_length));
//                rma_deb.trace(DEB_PREFIX("rma_receiver")
//                    , hpx::debug::mem_crc32(message, message_length, "Message region (recv rma)"));

                // do this after dumping out data as otherwise we lose some debug info
                std::cout << "ERROR " << message_region_->get_message_length() << " " << header_->message_size() << std::endl;
                HPX_ASSERT(message_region_->get_message_length() == header_->message_size());
            }
            else
            {
                HPX_ASSERT(header_->message_data());
                message = header_->message_data();
//                rma_deb.trace(DEB_PREFIX("rma_receiver")
//                    , hpx::debug::mem_crc32(message, message_length,
//                    "Message region (recv piggyback with rma)"));
            }

            for (auto &r : rma_regions_)
            {
                HPX_UNUSED(r);
//                rma_deb.trace(DEB_PREFIX("rma_receiver")
//                    , hpx::debug::mem_crc32(r->get_address(), r->get_message_length(),
//                    "rma region (recv) "));
            }

            // message reading is complete
            rma_deb.debug(DEB_PREFIX("rma_receiver") , "Sending ack");
            send_rma_complete_ack();

            rma_deb.debug(DEB_PREFIX("rma_receiver")
                , "message complete with RMA");

            if (user_recv_cb_) {
                user_recv_cb_();
            }

            cleanup_receive();
            return true;
        }

        // --------------------------------------------------------------------
        // Once all RMA reads are complete, we must send an ack to the origin
        // of the parcel so that it can release the RMA regions it is holding onto
        void send_rma_complete_ack()
        {
            ghex::rma_deb.scope(this, __func__);
            rma_deb.debug(DEB_PREFIX("rma_receiver")
                , "RMA Get owner" , hpx::debug::hex(header_->owner())
                , "has completed : injecting 8 byte ack to origin"
                , hpx::debug::dec<3>(src_addr_));

            ++sent_ack_;

//            region_type *wasted_region =
//                memory_pool_->allocate_region(sizeof(std::uint64_t));
//            void *desc[1] = { wasted_region->get_local_key() };
            std::uint64_t owner = this->header_->owner();
//            std::memcpy(wasted_region->get_address(), &owner, sizeof(std::uint64_t));

            bool ok = false;
            while(!ok) {
                // when we received the incoming message, the owner was already set
                // with the sender context so that we can signal it directly
                // that we have completed RMA and the sender my now cleanup.
                // Note : fi_inject does not trigger a completion locally, it just
                // sends and then we can reuse buffers and move on.
//                rma_deb.debug(DEB_PREFIX("rma_receiver")
//                    , "RMA Get owner" , hpx::debug::hex(header_->owner())
//                    , "has completed : fi_send 8 byte ack to origin"
//                    , hpx::debug::dec<3>(src_addr_));
//                ssize_t ret = fi_send(this->endpoint_, wasted_region->get_address(), sizeof(std::uint64_t), desc, this->src_addr_, this);
                ssize_t ret = fi_inject(this->endpoint_, &owner, sizeof(std::uint64_t), this->src_addr_);
                if (ret==0) {
                    ok = true;
                    rma_deb.debug(DEB_PREFIX("rma_receiver")
                        , "ack send ok"
                        , hpx::debug::dec<3>(src_addr_), hpx::debug::ptr(owner));
                }
                else if (ret == -FI_EAGAIN)
                {
                    rma_deb.error(DEB_PREFIX("rma_receiver")
                        , "reposting fi_inject...\n");
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
                else if (ret)
                {
                    throw fabric_error(ret, "fi_inject ack notification error");
                }
            }
//            memory_pool_->deallocate(wasted_region);
        }

        // --------------------------------------------------------------------
        // After message processing is complete, this routine cleans up and resets
        void cleanup_receive()
        {
            ghex::rma_deb.scope(this, __func__);
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
        // convenience function to execute a read for each zero-copy chunk
        // in the chunks_ variable
        void read_chunk_list()
        {
            ghex::rma_deb.scope(this, __func__);
            if (chunks_.size()==1)
            {
                // the rma info from the remote node is stored here
                chunktype &c = chunks_[0];

                // the rma info for the local node is here
                region_type *get_region = message_region_;

                // the local size should match the remote size
                rma_deb.debug(DEB_PREFIX("rma_receiver")
                    , "local rma size",  hpx::debug::hex<6>(get_region->get_message_length())
                    , "remote rma size", hpx::debug::hex<6>(c.size_));
                HPX_ASSERT(c.size_ == get_region->get_message_length());
                get_region->set_message_length(c.size_);
                rma_deb.debug(DEB_PREFIX("rma_receiver")
                    , "set message region size"
                    , *get_region);

                // save our region for cleanup later
                rma_regions_.push_back(get_region);

                // call the rma read function for the chunk
                const void *remote_addr = c.data_.cpos_;
//                c.data_.cpos_ = get_region->get_address();
                read_one_chunk(src_addr_, get_region, remote_addr, c.rma_);
            }
            else for (chunktype &c : chunks_)
            {
                if (c.type_ == detail::chunk_type_pointer ||
                    c.type_ == detail::chunk_type_rma)
                {
                    region_type *get_region =
                        memory_pool_->allocate_region(c.size_);
                    // Set the used space limit to the incoming buffer size
                    get_region->set_message_length(c.size_);

//                    rma_deb.trace(DEB_PREFIX("rma_receiver")
//                        , hpx::debug::mem_crc32(get_region->get_address(), c.size_,
//                            "(RMA GET region (new))"));

                    // store the remote key in case we overwrite it
                    std::uint64_t remote_key = c.rma_;

                    if (c.type_ == detail::chunk_type_rma) {
                        // rma object/vector chunks are not deleted
                        // so do not add them to the rma_regions list for cleanup
                        rma_deb.trace(DEB_PREFIX("chunk_type_rma")
                            , "Passing rma region to chunk structure");
                        c.rma_ = std::uintptr_t(get_region);
                    }
                    else {
                        rma_regions_.push_back(get_region);
                    }
                    // overwrite the serialization chunk data pointer because the chunk
                    // info sent contains the pointer to the remote data and when we
                    // decode the parcel we want the chunk to point to the local copy of it
                    const void *remote_addr = c.data_.cpos_;
                    c.data_.cpos_ = get_region->get_address();

                    // call the rma read function for the chunk
                    read_one_chunk(src_addr_, get_region, remote_addr, remote_key);
                }
            }
        }

        // --------------------------------------------------------------------
        // convenience function to execute a read, given the right params
        void read_one_chunk(
            fi_addr_t /*src_addr*/, region_type *get_region,
            const void *remote_addr, uint64_t rkey)
        {
            ghex::rma_deb.scope(this, __func__);
            // post the rdma read/get
            rma_deb.debug(DEB_PREFIX("rma_receiver")
                , "RMA Get fi_read"
                , "client" , hpx::debug::ptr(endpoint_)
                , "fi_addr" , hpx::debug::ptr(src_addr_)
                , "owner" , hpx::debug::ptr(header_->owner())
                , "local addr" , hpx::debug::ptr(get_region->get_address())
                , "local key" , hpx::debug::ptr(get_region->get_local_key())
                , "size" , hpx::debug::hex<6>(get_region->get_message_length())
                , "rkey" , hpx::debug::ptr(rkey)
                , "remote cpos" , hpx::debug::ptr(remote_addr));

            // count reads
            ++rma_reads_;

            bool ok = false;
            while (!ok) {
                if (rma_deb.is_enabled()) {
                    // write a pattern and dump out data for debugging purposes
                    uint32_t *buffer =
                        reinterpret_cast<uint32_t*>(get_region->get_address());
                    std::fill(buffer, buffer + get_region->get_size()/4,
                       0xDEADC0DE);
//                    rma_deb.trace(DEB_PREFIX("rma_receiver")
//                        , hpx::debug::mem_crc32(get_region->get_address(), get_region->get_message_length(),
//                                  "(RMA GET region (pre-fi_read))"));
                }

#ifdef EXPERIMENTAL_RMA_READ_COMPLETION
                // rma info of local buffer
                iovec local_regions[1] = {
                    get_region->get_address(),
                    get_region->get_message_length()
                };

                void *local_desc[1] = {
                    get_region->get_local_key()
                };

                // rma info of remote buffer
                fi_rma_iov remote_regions[1] = {
                    (uint64_t)(remote_addr),
                    get_region->get_message_length(),
                    rkey
                };

                fi_msg_rma rma_msg = {
                    local_regions,
                    local_desc,
                    1,
                    src_addr_,
                    remote_regions,
                    1,
                    this,
                    0
                };

                ssize_t ret = fi_readmsg(endpoint_, &rma_msg, FI_COMPLETION);
#else
                ssize_t ret = fi_read(endpoint_, get_region->get_address(),
                    get_region->get_message_length(), get_region->get_local_key(),
                    src_addr_, (uint64_t)(remote_addr), rkey, this);
#endif
                if (ret==0) {
                    ok = true;
                }
                else if (ret == -FI_EAGAIN)
                {
                    rma_deb.error(DEB_PREFIX("rma_receiver")
                        , "reposting fi_read...\n");
//                    parcelport_->background_work(0,
//                        hpx::parcelset::parcelport_background_mode_all);
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
                else if (ret)
                {
                    throw fabric_error(ret, "fi_read");
                }
            }
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
}}}

#endif
