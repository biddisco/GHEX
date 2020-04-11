#ifndef GHEX_LIBFABRIC_SENDER_HPP
#define GHEX_LIBFABRIC_SENDER_HPP

#include <ghex/transport_layer/libfabric/unique_function.hpp>
#include <ghex/transport_layer/libfabric/rma/detail/memory_region_impl.hpp>
#include <ghex/transport_layer/libfabric/rma/memory_pool.hpp>
#include <ghex/transport_layer/libfabric/rma/atomic_count.hpp>
//
#include <ghex/transport_layer/libfabric/print.hpp>
#include <ghex/transport_layer/libfabric/performance_counter.hpp>
//
#include <ghex/transport_layer/libfabric/libfabric_region_provider.hpp>
#include <ghex/transport_layer/libfabric/header.hpp>
#include <ghex/transport_layer/libfabric/controller.hpp>
#include <ghex/transport_layer/libfabric/rma_base.hpp>
#include <ghex/transport_layer/libfabric/rma_receiver.hpp>

#include <ghex/transport_layer/libfabric/locality.hpp>
#include <ghex/transport_layer/libfabric/print.hpp>
#include <ghex/transport_layer/callback_utils.hpp>
#include <ghex/transport_layer/message_buffer.hpp>

#include <boost/container/small_vector.hpp>
//
#include <memory>
// include for iovec
#include <sys/uio.h>

namespace ghex {
    // cppcheck-suppress ConfigurationNotChecked
    static hpx::debug::enable_print<false> send_deb("SENDER ");
}

namespace ghex {
namespace tl {
namespace libfabric
{

    typedef libfabric_region_provider                        region_provider;
    typedef rma::detail::memory_region_impl<region_provider> region_type;
    typedef rma::memory_pool<region_provider>                memory_pool_type;

    struct sender : public rma_base
    {
        typedef libfabric_region_provider                        region_provider;
        typedef rma::detail::memory_region_impl<region_provider> region_type;
        typedef rma::memory_pool<region_provider>                memory_pool_type;

        typedef header<GHEX_LIBFABRIC_MESSAGE_HEADER_SIZE> header_type;
        static constexpr unsigned int header_size = header_type::header_block_size;

        typedef boost::container::small_vector<region_type*,8> zero_copy_vector;

        // --------------------------------------------------------------------
        sender(controller* cnt, fid_ep* endpoint, fid_domain* domain,
            memory_pool_type* memory_pool)
          : rma_base(ctx_sender)
          , controller_(cnt)
          , endpoint_(endpoint)
          , domain_(domain)
          , memory_pool_(memory_pool)
          , dst_addr_(-1)
          , header_region_(nullptr)
          , chunk_region_(nullptr)
          , message_region_(nullptr)
          , message_region_temp_(false)
          , message_region_external_(false)
          , header_(nullptr)
          , completion_count_(0)
          , send_tag_(uint64_t(-1))
          , sends_posted_(0)
          , sends_deleted_(0)
          , acks_received_(0)
        {
            // the header region is reused multiple times
            header_region_ =
                memory_pool_->allocate_region(memory_pool_->small_.chunk_size());
            send_deb.debug(hpx::debug::str<>("Create sender")
                , hpx::debug::ptr(this));
//            user_handler_           = [](){};
//            postprocess_handler_    = [](sender*){};
//            handler_                = [](int const &){};
        }

        // --------------------------------------------------------------------
        ~sender()
        {
            auto scp = ghex::send_deb.scope(__func__);
            ghex::send_deb.debug(hpx::debug::str<>("map"), memory_pool_->region_alloc_pointer_map_.debug_map());
            memory_pool_->deallocate(header_region_);
        }

        // --------------------------------------------------------------------
//        snd_buffer_type get_new_buffer()
//        {
//            send_deb.debug("get_new_buffer"
//                , "Returning a new buffer object from sender"
//                , hpx::debug::ptr(this));
//            return snd_buffer_type(snd_data_type(memory_pool_), memory_pool_);
//        }

        // --------------------------------------------------------------------
        // The main message send routine : package the header, send it
        // with an optional extra message region if it cannot be piggybacked
        // send chunk/rma information for all zero copy serialization regions
        template<typename Callback>
        void async_send(const void *data,
                        std::size_t size,
                        uint64_t tag,
                        Callback &&cb_fn)
        {
            auto scp = ghex::send_deb.scope(__func__);
            ghex::send_deb.debug(hpx::debug::str<>("map"), memory_pool_->region_alloc_pointer_map_.debug_map());
            HPX_ASSERT(message_region_ == nullptr);
            HPX_ASSERT(completion_count_ == 0);

            // increment counter of total messages sent
            ++sends_posted_;
            int rma_chunks      = 0;
            send_tag_           = tag;
            user_send_cb_       = std::move(cb_fn);

            message_region_ = dynamic_cast<region_type*>(
                        memory_pool_->region_from_address(data));

            message_region_temp_ = false;
            if (message_region_ == nullptr) {
                message_region_temp_ = true;
                message_region_ = memory_pool_->register_temporary_region(data, size);
                memory_pool_->add_address_to_map(data, message_region_);
            }

            detail::chunktype ghex_chunk = detail::create_rma_chunk(
                message_region_->address_, size, message_region_->get_remote_key()
            );

            // reserve some space for zero copy information
            // ghex only uses one region at most (for now at least)
//            rma_regions_.reserve(1);
//            rma_chunks++;

            // create the header using placement new in the pinned memory block
            char *header_memory = (char*)(header_region_->get_address());

            send_deb.debug(hpx::debug::str<>("Placement new"));
            header_ = new(header_memory) header_type(ghex_chunk, this);
            header_region_->set_message_length(header_->header_length());
            send_deb.debug(hpx::debug::str<>("header"), *header_);

            // Get the block of pinned memory where the message resides
            message_region_->set_message_length(header_->message_size());
            HPX_ASSERT(header_->message_size() == size);

            send_deb.debug(hpx::debug::str<>("message buffer")
                , hpx::debug::ptr(data)
                , *message_region_);

            // The number of completions we need before cleaning up:
            // 1 (header block send) + 1 (ack message if we have RMA chunks)
            completion_count_ = 1;
            region_list_[0] = {
                header_region_->get_address(), header_region_->get_message_length() };
            region_list_[1] = {
                message_region_->get_address(), message_region_->get_message_length() };

            desc_[0] = header_region_->get_local_key();
            desc_[1] = message_region_->get_local_key();
            if (rma_regions_.size()>0 || rma_chunks>0 || !header_->message_piggy_back())
            {
                completion_count_ = 2;
            }

            if (header_->chunk_data()) {
                send_deb.debug(hpx::debug::str<>("Sender"), hpx::debug::ptr(this)
                    , "Chunk info is piggybacked");
            }
            else {
                send_deb.trace(hpx::debug::str<>("header-chunk rma")
                    , "zero-copy chunks" , hpx::debug::dec<>(rma_regions_.size())
                    , "rma chunks" , hpx::debug::dec<>(rma_chunks));
                auto &cb = header_->chunk_header_ptr()->chunk_rma;
                chunk_region_  = memory_pool_->allocate_region(cb.size_);
                cb.data_.pos_  = chunk_region_->get_address();
                cb.rma_        = chunk_region_->get_remote_key();
                send_deb.error("Not implemented chunk data copying");
//                std::memcpy(cb.data_.pos_, buffer_.chunks_.data(), cb.size_);
                send_deb.debug("Set up header-chunk rma data with"
                    , "size"   , hpx::debug::dec<>(cb.size_)
                    , "rma"    , hpx::debug::ptr(cb.rma_)
                    , "addr"   , hpx::debug::ptr(cb.data_.cpos_));
            }

            if (header_->message_piggy_back())
            {
                send_deb.debug(hpx::debug::str<>("Sender"), hpx::debug::ptr(this)
                    , "tag", hpx::debug::hex<16>(tag)
                    , "Main message is piggybacked");

//                send_deb.trace(hpx::debug::str<>("Sender"), hpx::debug::ptr(this)
//                    , hpx::debug::mem_crc32(header_region_->get_address()
//                    , header_region_->get_message_length()
//                    , "Header region (send piggyback)"));

//                send_deb.trace(hpx::debug::str<>("Sender"), hpx::debug::ptr(this)
//                    , hpx::debug::mem_crc32(message_region_->get_address()
//                    , message_region_->get_message_length()
//                    , "Message region (send piggyback)"));

                // send 2 regions as one message, goes into one receive
                bool ok = false;
                while (!ok) {
                    HPX_ASSERT(
                        (this->region_list_[0].iov_len + this->region_list_[1].iov_len) <=
                            GHEX_LIBFABRIC_MESSAGE_HEADER_SIZE);
                    ssize_t ret;
                    if (send_tag_==uint64_t(-1)) {
                        ret = fi_sendv(this->endpoint_, this->region_list_,
                            this->desc_, 2, this->dst_addr_, this);
                    }
                    else {
                        ret = fi_tsendv(this->endpoint_, this->region_list_,
                            this->desc_, 2, this->dst_addr_, this->send_tag_, this);
                    }

                    if (ret == 0) {
                        ok = true;
                    }
                    else if (ret == -FI_EAGAIN) {
                        send_deb.error("Reposting fi_sendv / fi_tsendv");
//                        controller_->background_work(0, controller_background_mode_all);
                    }
                    else if (ret == -FI_ENOENT) {
                        // if a node has failed, we can recover @TODO : put something here
                        send_deb.error("No destination endpoint, retrying after 1s ...");
                        std::terminate();
                    }
                    else if (ret)
                    {
                        throw fabric_error(int(ret), "fi_sendv / fi_tsendv");
                    }
                }
            }
            else
            {
                header_->set_message_rdma_info(
                    message_region_->get_remote_key(), message_region_->get_address());

                send_deb.debug(hpx::debug::str<>("Sender"), hpx::debug::ptr(this)
                    , "tag", hpx::debug::hex<16>(tag)
                    , "message region NOT piggybacked"
                    , hpx::debug::hex<6>(size)
                    , *message_region_);

//                send_deb.trace(hpx::debug::str<>("Sender"), hpx::debug::ptr(this)
//                    , hpx::debug::mem_crc32(header_region_->get_address()
//                    , header_region_->get_message_length()
//                    , "Header region (pre-send)"));

//                send_deb.trace(hpx::debug::str<>("Sender"), hpx::debug::ptr(this)
//                    , hpx::debug::mem_crc32(message_region_->get_address()
//                    , message_region_->get_message_length()
//                    , "Message region (send for rma fetch)"));

                bool ok = false;
                while (!ok) {
                    ssize_t ret;
                    if (send_tag_==uint64_t(-1)) {
                        ret = fi_send(this->endpoint_,
                                      this->region_list_[0].iov_base,
                                      this->region_list_[0].iov_len,
                                      this->desc_[0], this->dst_addr_, this);
                    }
                    else {
                        ret = fi_tsend(this->endpoint_,
                                       this->region_list_[0].iov_base,
                                       this->region_list_[0].iov_len,
                                       this->desc_[0], this->dst_addr_, send_tag_, this);
                    }

                    if (ret == 0) {
                        ok = true;
                    }
                    else if (ret == -FI_EAGAIN)
                    {
                        send_deb.error("reposting fi_send / fi_tsend");
//                        controller_->background_work(0, controller_background_mode_all);
                    }
                    else if (ret)
                    {
                        throw fabric_error(int(ret), "fi_send / fi_tsend");
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // The main message send routine : package the header, send it
        // with an optional extra message region if it cannot be piggybacked
        // send chunk/rma information for all zero copy serialization regions
        template<typename Message, typename Callback>
        void async_send(const Message& msg,
                        uint64_t tag,
                        Callback &&cb_fn)
        {
            async_send(msg.data(), msg.size(), tag, std::forward<Callback>(cb_fn));       
        }

        // --------------------------------------------------------------------
        // Called when a send completes
        int handle_send_completion()
        {
            auto scp = ghex::send_deb.scope(__func__);
            ghex::send_deb.debug(hpx::debug::str<>("map"), memory_pool_->region_alloc_pointer_map_.debug_map());
            send_deb.debug(hpx::debug::str<>("Sender"), hpx::debug::ptr(this)
                , "handle_send_completion"
                , "tag", hpx::debug::hex<16>(send_tag_)
                , "RMA regions" , hpx::debug::dec<>(rma_regions_.size())
                , "completion count" , hpx::debug::dec<>(completion_count_));
            return cleanup();
        }

        // --------------------------------------------------------------------
        // Triggered when the remote end has finished RMA operations and
        // we can release resources
        int handle_message_completion_ack()
        {
            auto scp = ghex::send_deb.scope(__func__);
            ghex::send_deb.debug(hpx::debug::str<>("map"), memory_pool_->region_alloc_pointer_map_.debug_map());
            send_deb.debug(hpx::debug::str<>("Sender"), hpx::debug::ptr(this)
                , "handle_message_completion_ack"
                , "RMA regions" , hpx::debug::dec<>(rma_regions_.size())
                , "completion count" , hpx::debug::dec<>(completion_count_));
            ++acks_received_;
            return cleanup();
        }

        // --------------------------------------------------------------------
        // Cleanup memory regions we are holding onto etc
        // returns 0 if the cleanup terminates because more message parts are needed
        // returns 1 if the cleanup succeeds as the message is complete and the
        // user callback is triggered
        int cleanup()
        {
            auto scp = ghex::send_deb.scope(__func__);
            ghex::send_deb.debug(hpx::debug::str<>("map"), memory_pool_->region_alloc_pointer_map_.debug_map());
            send_deb.debug(hpx::debug::str<>("Sender"), hpx::debug::ptr(this)
                , "decrementing completion_count from", hpx::debug::dec<>(completion_count_));

            // if we need to wait for more completion events, return without cleaning
            if (--completion_count_ > 0)
                return 0;

            // track deletions
            ++sends_deleted_;

            user_send_cb_();
            user_send_cb_ = [](){};

            // cleanup header and message region
            send_deb.debug(hpx::debug::str<>("Sender"), hpx::debug::ptr(this)
                , "disposing of message region");
            if (!message_region_external_) {
                if (!message_region_temp_) {
                    memory_pool_->deallocate(message_region_);
                }
                else {
                    memory_pool_->deallocate(message_region_);
                }
            }

            message_region_ = nullptr;
            header_         = nullptr;

            // cleanup chunk region
            send_deb.debug(hpx::debug::str<>("Sender"), hpx::debug::ptr(this)
                , "disposing of chunk region");
            if (chunk_region_) {
                memory_pool_->deallocate(chunk_region_);
                chunk_region_ = nullptr;
            }

            send_deb.debug(hpx::debug::str<>("Sender"), hpx::debug::ptr(this)
                , "disposing of rma region(s)");
            for (auto& region: rma_regions_) {
                memory_pool_->deallocate(region);
            }
            rma_regions_.clear();
            send_deb.debug(hpx::debug::str<>("Sender"), hpx::debug::ptr(this)
                , "calling postprocess_handler");
            if (postprocess_handler_) postprocess_handler_(this);
            send_deb.debug(hpx::debug::str<>("Sender"), hpx::debug::ptr(this)
                , "completed cleanup/postprocess_handler");
            return 1;
        }

        // --------------------------------------------------------------------
        // if a send completion reports failure, we can retry the send
        void handle_error(struct fi_cq_err_entry /*err*/)
        {
            send_deb.error(hpx::debug::str<>("resending"), hpx::debug::ptr(this));

            if (header_->message_piggy_back())
            {
                // send 2 regions as one message, goes into one receive
                bool ok = false;
                while (!ok) {
                    HPX_ASSERT(
                        (this->region_list_[0].iov_len + this->region_list_[1].iov_len) <=
                            GHEX_LIBFABRIC_MESSAGE_HEADER_SIZE);
                    ssize_t ret = fi_sendv(this->endpoint_, this->region_list_,
                        this->desc_, 2, this->dst_addr_, this);

                    if (ret == 0) {
                        ok = true;
                    }
                    else if (ret == -FI_EAGAIN)
                    {
                        send_deb.error("reposting fi_sendv...\n");
//                        controller_->background_work(0, controller_background_mode_all);
                    }
                    else if (ret)
                    {
                        throw fabric_error(int(ret), "fi_sendv");
                    }
                }
            }
            else
            {
                header_->set_message_rdma_info(
                    message_region_->get_remote_key(), message_region_->get_address());

                // send just the header region - a single message
                bool ok = false;
                while (!ok) {
                    ssize_t ret = fi_send(this->endpoint_,
                        this->region_list_[0].iov_base,
                        this->region_list_[0].iov_len,
                        this->desc_[0], this->dst_addr_, this);

                    if (ret == 0) {
                        ok = true;
                    }
                    else if (ret == -FI_EAGAIN)
                    {
                        send_deb.error("reposting fi_send...\n");
//                        controller_->background_work(0, controller_background_mode_all);
                    }
                    else if (ret)
                    {
                        throw fabric_error(int(ret), "fi_sendv");
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // print out some info that is useful
        friend std::ostream & operator<<(std::ostream & os, const sender &s)
        {
            if (s.header_) {
                os << "sender " << hpx::debug::ptr(&s) << "header block " << *(s.header_);
            }
            else {
                os << "sender " << hpx::debug::ptr(&s) << "header block nullptr";
            }
            return os;
        }

        // --------------------------------------------------------------------
        controller                  *controller_;
        fid_ep                      *endpoint_;
        fid_domain                  *domain_;
        memory_pool_type            *memory_pool_;
        fi_addr_t                    dst_addr_;
        region_type                 *header_region_;
        region_type                 *chunk_region_;
        region_type                 *message_region_;
        bool                         message_region_temp_;
        bool                         message_region_external_;
        header_type                 *header_;
        zero_copy_vector             rma_regions_;
        hpx::util::atomic_count      completion_count_;
        uint64_t                     send_tag_;
        unique_function<void(void)>  user_send_cb_;
        std::function<void(sender*)> postprocess_handler_;

        // principally for debugging
        performance_counter<unsigned int> sends_posted_;
        performance_counter<unsigned int> sends_deleted_;
        performance_counter<unsigned int> acks_received_;
        //
        //
        struct iovec region_list_[2];
        void*        desc_[2];
    };
}}}

#endif
