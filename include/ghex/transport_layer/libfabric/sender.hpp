#ifndef GHEX_LIBFABRIC_SENDER_HPP
#define GHEX_LIBFABRIC_SENDER_HPP

#include <ghex/transport_layer/libfabric/unique_function.hpp>
#include <ghex/transport_layer/libfabric/rma/detail/memory_region_impl.hpp>
#include <ghex/transport_layer/libfabric/rma/memory_pool.hpp>
#include <ghex/transport_layer/libfabric/rma/detail/memory_region_allocator.hpp>
#include <ghex/transport_layer/libfabric/rma/atomic_count.hpp>
//
#include <ghex/transport_layer/libfabric/print.hpp>
#include <ghex/transport_layer/libfabric/performance_counter.hpp>
//
#include <ghex/transport_layer/libfabric/libfabric_region_provider.hpp>
#include <ghex/transport_layer/libfabric/rma/memory_pool.hpp>
#include <ghex/transport_layer/libfabric/controller.hpp>
#include <ghex/transport_layer/libfabric/rma_base.hpp>

#include <ghex/transport_layer/libfabric/locality.hpp>
#include <ghex/transport_layer/libfabric/print.hpp>
#include <ghex/transport_layer/callback_utils.hpp>
#include <ghex/transport_layer/message_buffer.hpp>
//
#include <memory>
// include for iovec
#include <sys/uio.h>

namespace gridtools { namespace ghex {
    // cppcheck-suppress ConfigurationNotChecked
    static hpx::debug::enable_print<false> send_deb("SENDER ");
}}

namespace gridtools {
namespace ghex {
namespace tl {
namespace libfabric
{
    struct sender : public rma_base
    {
        using region_provider    = libfabric_region_provider;
        using region_type        = rma::detail::memory_region_impl<region_provider>;
        using memory_pool_type   = rma::memory_pool<region_provider>;
        using libfabric_msg_type = message_buffer<rma::memory_region_allocator<unsigned char>>;
        using any_msg_type       = gridtools::ghex::tl::cb::any_message;

        // --------------------------------------------------------------------
        sender(controller* cnt, fid_ep* endpoint, fid_domain* domain,
            memory_pool_type* memory_pool)
          : rma_base(ctx_sender)
          , controller_(cnt)
          , endpoint_(endpoint)
          , domain_(domain)
          , memory_pool_(memory_pool)
          , dst_addr_(-1)
          , message_region_(nullptr)
          , message_region_temp_(false)
          , message_region_mapped_(false)
          , tag_(uint64_t(-1))
          , sends_posted_(0)
          , sends_deleted_(0)
        {
        }

        // --------------------------------------------------------------------
        ~sender()
        {
            [[maybe_unused]] auto scp = ghex::send_deb.scope(__func__);
        }        

        // --------------------------------------------------------------------
        // this takes a pinned memory region and sends it
        void send_tagged_region(region_type *send_region)
        {
            [[maybe_unused]] auto scp = ghex::send_deb.scope(__func__);
            ghex::send_deb.debug(hpx::debug::str<>("map contents")
                                 , memory_pool_->region_alloc_pointer_map_.debug_map());

            // increment counter of total messages sent
            ++sends_posted_;

            send_deb.debug(hpx::debug::str<>("message buffer"), *send_region);

            bool ok = false;
            while (!ok) {
                ssize_t ret;
                ret = fi_tsend(this->endpoint_,
                               send_region->get_address(),
                               send_region->get_message_length(),
                               send_region->get_local_key(),
                               this->dst_addr_, tag_, this);
                if (ret == 0) {
                    ok = true;
                }
                else if (ret == -FI_EAGAIN) {
                    send_deb.error("Reposting fi_sendv / fi_tsendv");
                    // controller_->background_work(0, controller_background_mode_all);
                }
                else if (ret == -FI_ENOENT) {
                    // if a node has failed, we can recover
                    // @TODO : put something better here
                    send_deb.error("No destination endpoint, terminating.");
                    std::terminate();
                }
                else if (ret)
                {
                    throw fabric_error(int(ret), "fi_sendv / fi_tsendv");
                }
            }
        }

        // --------------------------------------------------------------------
        // Take raw data and send it.
        // The data might not be pinned already, if not, pin it temporarily
        void send_tagged_data(const void *data,
                        std::size_t size)
        {
            [[maybe_unused]] auto scp = ghex::send_deb.scope(__func__);

            // did someone register this memory block and store it in the memory pool map
            message_region_ = dynamic_cast<region_type*>(
                        memory_pool_->region_from_address(data));

            // if the memory was not pinned, register it now
            message_region_temp_ = false;
            if (message_region_ == nullptr) {
                message_region_temp_ = true;
                message_region_ = memory_pool_->register_temporary_region(data, size);
                memory_pool_->add_address_to_map(data, message_region_);
                ghex::send_deb.debug(hpx::debug::str<>("region is temp"), message_region_);
            }

            // Set the used size correctly
            message_region_->set_message_length(size);
            send_tagged_region(message_region_);
        }

        // utility struct to hold raw data info
        struct msg_data_default {
            const void *data;
            std::size_t size;
        };

        // utility struct to hold libfabric enabled info
        struct msg_data_libfabric {
            region_type *message_region_;
        };

        // solve move/callback issues by extracting what we need from the message
        // in this function, before calling the main send function after the
        // message has been moved into the callback
        auto init_message_data(const any_msg_type &msg, uint64_t tag)
        {
            tag_     = tag;
            return msg_data_default{msg.data(), msg.size()};
        }

        auto init_message_data(const libfabric_msg_type &msg, uint64_t tag)
        {
            tag_                 = tag;
            message_region_temp_ = false;
            message_region_      = msg.get_buffer().m_pointer.region_;
            message_region_->set_message_length(msg.size());
            return msg_data_libfabric{message_region_};
        }

        // generic message sender (reference to message)
        template<typename Message, typename Callback>
        void send_tagged_msg(const Message &msg,
                             uint64_t tag,
                             Callback &&cb_fn)
        {
            tag_     = tag;
            user_cb_ = std::move(cb_fn);
            send_tagged_data(msg.data(), msg.size());
        }

        template<typename Callback>
        void send_tagged_msg(libfabric_msg_type &msg,
                             uint64_t tag,
                             Callback &&cb_fn)
        {
            init_message_data(msg, tag);
            user_cb_ = std::move(cb_fn);
            send_tagged_region(message_region_);
        }

        // generic message sender (move message into callback)
        template<typename Callback>
        void send_tagged_msg(const msg_data_default &md, Callback &&cb_fn)
        {
            user_cb_ = std::forward<Callback>(cb_fn);
            send_tagged_data(md.data, md.size);
        }

        // libfabric message customization (known memory region)
        template<typename Callback>
        void send_tagged_msg(const msg_data_libfabric &md, Callback &&cb_fn)
        {
            user_cb_ = std::forward<Callback>(cb_fn);
            send_tagged_region(md.message_region_);
        }

        // --------------------------------------------------------------------
        // Called when a send completes
        int handle_send_completion()
        {
            [[maybe_unused]] auto scp = ghex::send_deb.scope(__func__);
            ghex::send_deb.debug(hpx::debug::str<>("map contents")
                                 , memory_pool_->region_alloc_pointer_map_.debug_map());

            send_deb.debug(hpx::debug::str<>("Sender"), hpx::debug::ptr(this)
                , "handle_send_completion"
                , "tag", hpx::debug::hex<16>(tag_));

            // track deletions
            ++sends_deleted_;

            // invoke the user supplied callback
            user_cb_();

            // clear the callback to an empty state
            // (in case it holds reference counts that must be released)
            user_cb_ = [](){};

            // cleanup temp region
            if (message_region_temp_) {
                send_deb.debug(hpx::debug::str<>("Sender")
                               , hpx::debug::ptr(this)
                               , "disposing of temp region");
                ghex::send_deb.debug(hpx::debug::str<>("free temp region "), message_region_);
                memory_pool_->remove_address_from_map(message_region_->get_address(), message_region_);
                memory_pool_->deallocate(message_region_);
                ghex::send_deb.debug(hpx::debug::str<>("map contents")
                                     , memory_pool_->region_alloc_pointer_map_.debug_map());
            }
            message_region_ = nullptr;

            // return the sender to the available list
            send_deb.debug(hpx::debug::str<>("Sender")
                           , hpx::debug::ptr(this)
                           , "calling postprocess_handler");
            postprocess_handler_(this);

            return 1;
        }

        // --------------------------------------------------------------------
        // if a send completion reports failure, we can retry the send
        void handle_error(struct fi_cq_err_entry /*err*/)
        {
            send_deb.error(hpx::debug::str<>("resending"), hpx::debug::ptr(this));

            throw fabric_error(0, "Should resend message here");
            std::terminate();
        }

        // --------------------------------------------------------------------
        controller                  *controller_;
        fid_ep                      *endpoint_;
        fid_domain                  *domain_;
        memory_pool_type            *memory_pool_;
        fi_addr_t                    dst_addr_;
        region_type                 *message_region_;
        bool                         message_region_temp_;
        bool                         message_region_mapped_;
        uint64_t                     tag_;
        unique_function<void(void)>  user_cb_;
        std::function<void(sender*)> postprocess_handler_;

        // principally for debugging
        performance_counter<unsigned int> sends_posted_;
        performance_counter<unsigned int> sends_deleted_;
        //
        struct iovec region_list_[2];
        void*        desc_[2];
    };
}}}}

#endif
