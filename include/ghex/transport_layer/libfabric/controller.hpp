#ifndef GHEX_LIBFABRIC_CONTROLLER_HPP
#define GHEX_LIBFABRIC_CONTROLLER_HPP

#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
// ??
#include <sstream>
#include <cstdint>
#include <cstddef>
#include <cstring>
//
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include "fabric_error.hpp"
//
#include <ghex/transport_layer/libfabric/print.hpp>
#include <ghex/transport_layer/libfabric/locality.hpp>
#include <ghex/transport_layer/libfabric/libfabric_region_provider.hpp>
#include <ghex/transport_layer/libfabric/rma/memory_pool.hpp>
#include <ghex/transport_layer/libfabric/receiver_fwd.hpp>
#include <ghex/transport_layer/libfabric/sender.hpp>
#include <ghex/transport_layer/libfabric/rma/detail/allocator.hpp>
//
#include <mpi.h>

// ------------------------------------------------
// underlying provider used by libfabric
// ------------------------------------------------
//#define GHEX_LIBFABRIC_PROVIDER_GNI 1
#define GHEX_LIBFABRIC_PROVIDER_SOCKETS

// ------------------------------------------------
// if we have PMI we can bootstrap
// ------------------------------------------------
//#define GHEX_LIBFABRIC_HAVE_PMI 1

// ------------------------------------------------
// implement bootstrapping directly using libfabric
// currently only valid with sockets provider
// ------------------------------------------------
// #define GHEX_LIBFABRIC_HAVE_BOOTSTRAPPING

// ------------------------------------------------
// Reliable Data Endpoint type
// ------------------------------------------------
#define GHEX_LIBFABRIC_ENDPOINT_RDM

// ------------------------------------------------
// Needed on Cray for GNI extensions
// ------------------------------------------------
#ifdef GHEX_LIBFABRIC_PROVIDER_GNI
# include "rdma/fi_ext_gni.h"
#endif

#ifdef GHEX_LIBFABRIC_HAVE_PMI
# include <pmi2.h>
#endif

#define GHEX_LIBFABRIC_FI_VERSION_MAJOR 1
#define GHEX_LIBFABRIC_FI_VERSION_MINOR 9

namespace ghex {
    // cppcheck-suppress ConfigurationNotChecked
    static hpx::debug::enable_print<true> cnt_deb("CONTROL");

#undef FUNC_START_DEBUG_MSG
#undef FUNC_END_DEBUG_MSG
#define FUNC_START_DEBUG_MSG ghex::cnt_deb.debug(hpx::debug::str<>("*** Enter ") , __func__);
#define FUNC_END_DEBUG_MSG   ghex::cnt_deb.debug(hpx::debug::str<>("### Exit  ") , __func__);
}

namespace ghex {
namespace tl {
namespace libfabric
{

//    template<typename Tag, typename ThreadPrimitives>
//    struct transport_context;

    class controller
    {
    public:
        typedef std::mutex                   mutex_type;
        typedef std::lock_guard<mutex_type>  scoped_lock;

        struct fi_info    *fabric_info_;
        struct fid_fabric *fabric_;
        struct fid_domain *fabric_domain_;
        // Server/Listener for RDMA connections.
        struct fid_ep     *ep_active_;
        struct fid_ep     *ep_shared_rx_cxt_;

        // we will use just one event queue for all connections
        struct fid_eq     *event_queue_;
        struct fid_cq     *txcq_, *rxcq_;
        struct fid_av     *av_;

        bool                       immediate_;
        std::atomic<std::uint32_t> bootstrap_counter_;

        // We must maintain a list of senders that are being used
        using sender_list =
            boost::lockfree::stack<
                sender*,
                boost::lockfree::capacity<GHEX_LIBFABRIC_MAX_SENDS>,
                boost::lockfree::fixed_sized<true>
            >;
        sender_list senders_;
        std::atomic<unsigned int> senders_in_use_;

        int      m_rank_;
        int      m_size_;
        MPI_Comm m_comm_;

        locality here_;
        locality root_;

        std::size_t get_num_localities() { return m_size_; }

        // --------------------------------------------------------------------
        // constructor gets info from device and sets up all necessary
        // maps, queues and server endpoint etc
        controller(
            std::string const &provider,
            std::string const &domain,
            std::string const &endpoint, bool bootstrap,
            int rank, int size, MPI_Comm mpi_comm)
          : fabric_info_(nullptr)
          , fabric_(nullptr)
          , fabric_domain_(nullptr)
          , ep_active_(nullptr)
          , ep_shared_rx_cxt_(nullptr)
          , event_queue_(nullptr)
            //
          , txcq_(nullptr)
          , rxcq_(nullptr)
          , av_(nullptr)
            //
          , immediate_(false)
          , senders_in_use_(0)
          , m_rank_(rank)
          , m_size_(size)
          , m_comm_(mpi_comm)

        {
            std::cout.setf(std::ios::unitbuf);
            FUNC_START_DEBUG_MSG;
            open_fabric(provider, domain, endpoint);

            // setup a passive listener, or an active RDM endpoint
            here_ = create_local_endpoint();
            cnt_deb.debug(hpx::debug::str<>("Overriding here") , iplocality(here_));

            // Create a memory pool for pinned buffers
            memory_pool_.reset(
                new rma::memory_pool<libfabric_region_provider>(fabric_domain_));

            // initialize rma allocator with a pool
            ghex::tl::libfabric::rma::allocator<unsigned char> allocator{};
            allocator.init_memory_pool(memory_pool_.get());

            initialize_localities();

#if defined(GHEX_LIBFABRIC_HAVE_BOOTSTRAPPING)
            if (bootstrap) {
#if defined(GHEX_LIBFABRIC_PROVIDER_SOCKETS)
                cnt_deb.debug(hpx::debug::str<>("Calling boot SOCKETS"));
                boot_SOCKETS();
#elif defined(GHEX_LIBFABRIC_HAVE_PMI)
                cnt_deb.debug("Calling boot PMI");
                boot_PMI();
# endif
                if (root_ == here_) {
                    std::cout << "Libfabric Parcelport boot-step complete" << std::endl;
                }
            }
#endif
            FUNC_END_DEBUG_MSG;
        }

        // --------------------------------------------------------------------
        // clean up all resources
        ~controller()
        {
            unsigned int messages_handled_ = 0;
            unsigned int acks_received_    = 0;
            unsigned int msg_plain_        = 0;
            unsigned int msg_rma_          = 0;
            unsigned int sent_ack_         = 0;
            unsigned int rma_reads_        = 0;
            unsigned int recv_deletes_     = 0;
            //
            for (auto &r : receivers_) {
                r.cleanup();
                // from receiver
                messages_handled_ += r.messages_handled_;
                acks_received_    += r.acks_received_;
            }

            rma_receiver *rcv = nullptr;
            while (receiver::rma_receivers_.pop(rcv))
            {
                msg_plain_    += rcv->msg_plain_;
                msg_rma_      += rcv->msg_rma_;
                sent_ack_     += rcv->sent_ack_;
                rma_reads_    += rcv->rma_reads_;
                recv_deletes_ += rcv->recv_deletes_;
                delete rcv;
            }

            cnt_deb.debug(hpx::debug::str<>("counters")
                , "Received messages"  , hpx::debug::dec<>(messages_handled_)
                , "Received acks "     , hpx::debug::dec<>(acks_received_)
                , "Sent acks "         , hpx::debug::dec<>(sent_ack_)
                , "Total reads "       , hpx::debug::dec<>(rma_reads_)
                , "Total deletes "     , hpx::debug::dec<>(recv_deletes_)
                , "deletes error "     , hpx::debug::dec<>(messages_handled_ - recv_deletes_));

            // Cleaning up receivers to avoid memory leak errors.
            receivers_.clear();

            cnt_deb.debug(hpx::debug::str<>("closing"), "fabric");
            if (fabric_)
                fi_close(&fabric_->fid);
#ifdef GHEX_LIBFABRIC_ENDPOINT_RDM
            cnt_deb.debug(hpx::debug::str<>("closing"), "ep_active");
            if (ep_active_)
                fi_close(&ep_active_->fid);
#endif
            cnt_deb.debug(hpx::debug::str<>("closing"), "event_queue");
            if (event_queue_)
                fi_close(&event_queue_->fid);
            cnt_deb.debug(hpx::debug::str<>("closing"), "fabric_domain");
            if (fabric_domain_)
                fi_close(&fabric_domain_->fid);
            cnt_deb.debug(hpx::debug::str<>("closing"), "ep_shared_rx_cxt");
            if (ep_shared_rx_cxt_)
                fi_close(&ep_shared_rx_cxt_->fid);
            // clean up
            cnt_deb.debug(hpx::debug::str<>("freeing fabric_info"));
            fi_freeinfo(fabric_info_);
        }

        // --------------------------------------------------------------------
        // boot all ranks when using libfabric sockets provider
        void boot_SOCKETS()
        {
#ifdef GHEX_LIBFABRIC_PROVIDER_SOCKETS
            // we expect N-1 localities to connect to us during bootstrap
            std::size_t N = get_num_localities();
            bootstrap_counter_ = N-1;

            cnt_deb.debug(hpx::debug::str<>("Bootstrap")
                , N , " localities");

            // create address vector and queues we need if bootstrapping
            create_completion_queues(fabric_info_, N);

            cnt_deb.debug(hpx::debug::str<>("inserting root addr")
                          , iplocality(root_));
            root_ = insert_address(root_);
#endif
        }

        // --------------------------------------------------------------------
        // send rank 0 address to root and receive array of localities
        void MPI_exchange_localities()
        {
            std::vector<char> localities(m_size_*locality_defs::array_size, 0);
            //
            if (m_rank_ > 0)
            {
                cnt_deb.debug(hpx::debug::str<>("sending here")
                    , iplocality(here_)
                    , "size", locality_defs::array_size);
                int err = MPI_Send(
                            here_.fabric_data(),
                            locality_defs::array_size,
                            MPI_CHAR,
                            0, // dst rank
                            0, // tag
                            m_comm_);

                cnt_deb.debug(hpx::debug::str<>("receiving all")
                    , "size", locality_defs::array_size);

                MPI_Status status;
                err = MPI_Recv(
                            localities.data(),
                            m_size_*locality_defs::array_size,
                            MPI_CHAR,
                            0, // src rank
                            0, // tag
                            m_comm_,
                            &status);
                cnt_deb.debug(hpx::debug::str<>("received addresses"));
            }
            else {
                cnt_deb.debug(hpx::debug::str<>("receiving addresses"));
                memcpy(&localities[0], here_.fabric_data(), locality_defs::array_size);
                for (int i=1; i<m_size_; ++i) {
                    cnt_deb.debug(hpx::debug::str<>("receiving address"), hpx::debug::dec<>(i));
                    MPI_Status status;
                    int err = MPI_Recv(
                                &localities[i*locality_defs::array_size],
                                m_size_*locality_defs::array_size,
                                MPI_CHAR,
                                i, // src rank
                                0, // tag
                                m_comm_,
                                &status);
                    cnt_deb.debug(hpx::debug::str<>("received address"), hpx::debug::dec<>(i));
                }

                cnt_deb.debug(hpx::debug::str<>("sending all"));
                for (int i=1; i<m_size_; ++i) {
                    cnt_deb.debug(hpx::debug::str<>("sending to"), hpx::debug::dec<>(i));
                    int err = MPI_Send(
                                &localities[0],
                                m_size_*locality_defs::array_size,
                                MPI_CHAR,
                                i, // dst rank
                                0, // tag
                                m_comm_);
                }
            }

            // all ranks should now have a full localities vector
            cnt_deb.debug(hpx::debug::str<>("populating vector"));
            for (int i=0; i<m_size_; ++i) {
                locality temp;
                int offset = i*locality_defs::array_size;
                memcpy(temp.fabric_data_writable(), &localities[offset], locality_defs::array_size);
                insert_address(temp);
            }
        }

        // --------------------------------------------------------------------
        // initialize the basic fabric/domain/name
        void open_fabric(std::string const& provider, std::string const& domain,
            std::string const& endpoint_type)
        {
            FUNC_START_DEBUG_MSG;
            struct fi_info *fabric_hints_ = fi_allocinfo();
            if (!fabric_hints_) {
                throw fabric_error(-1, "Failed to allocate fabric hints");
            }
            // we require message and RMA support, so ask for them
            // we also want receives to carry source address info
            here_ = locality("127.0.0.1", "7909");
            cnt_deb.debug(hpx::debug::str<>("Here locality") , iplocality(here_));

//#ifdef GHEX_LIBFABRIC_HAVE_BOOTSTRAPPING
#if defined(GHEX_LIBFABRIC_SOCKETS)
            // If we are the root node, then create connection with the right port address
            if (m_rank_ == 0) {
                cnt_deb.debug(hpx::debug::str<>("root locality = src")
                              , iplocality(root_));
                // this memory will (should) be deleted in hints destructor
                struct sockaddr_in *socket_data1 = new struct sockaddr_in();
                memcpy(socket_data1, here_.fabric_data(), locality_defs::array_size);
                fabric_hints_->addr_format  = FI_SOCKADDR_IN;
                fabric_hints_->src_addr     = socket_data1;
                fabric_hints_->src_addrlen  = sizeof(struct sockaddr_in);
            }
            else {
                cnt_deb.debug(hpx::debug::str<>("root locality = dest")
                              , iplocality(root_));
                // this memory will (should) be deleted in hints destructor
                struct sockaddr_in *socket_data2 = new struct sockaddr_in();
                memcpy(socket_data2, here_.fabric_data(), locality_defs::array_size);
                fabric_hints_->addr_format  = FI_SOCKADDR_IN;
                fabric_hints_->dest_addr    = socket_data2;
                fabric_hints_->dest_addrlen = sizeof(struct sockaddr_in);
            }
            //
            fabric_hints_->caps        = FI_MSG | FI_RMA | FI_SOURCE | /*FI_SOURCE_ERR |*/
                FI_WRITE | FI_READ | FI_REMOTE_READ | FI_REMOTE_WRITE | FI_RMA_EVENT;
#elif defined(GHEX_LIBFABRIC_GNI)
            fabric_hints_->caps                   = FI_MSG | FI_RMA | FI_SOURCE |
                FI_WRITE | FI_READ | FI_REMOTE_READ | FI_REMOTE_WRITE | FI_RMA_EVENT;
#endif
            fabric_hints_->mode                   = FI_CONTEXT | FI_LOCAL_MR;
            fabric_hints_->fabric_attr->prov_name = strdup(provider.c_str());
            cnt_deb.debug(hpx::debug::str<>("fabric provider")
                , fabric_hints_->fabric_attr->prov_name);
            if (domain.size()>0) {
                fabric_hints_->domain_attr->name  = strdup(domain.c_str());
                cnt_deb.debug(hpx::debug::str<>("fabric domain")
                    , fabric_hints_->domain_attr->name);
            }

            // use infiniband type basic registration for now
            fabric_hints_->domain_attr->mr_mode = FI_MR_BASIC;

            // Disable the use of progress threads
            fabric_hints_->domain_attr->control_progress = FI_PROGRESS_MANUAL;
            fabric_hints_->domain_attr->data_progress = FI_PROGRESS_MANUAL;

            // Enable thread safe mode (Does not work with psm2 provider)
            fabric_hints_->domain_attr->threading = FI_THREAD_SAFE;

            // Enable resource management
            fabric_hints_->domain_attr->resource_mgmt = FI_RM_ENABLED;

#ifdef GHEX_LIBFABRIC_ENDPOINT_RDM
            cnt_deb.debug(hpx::debug::str<>("fabric endpoint"), "RDM");
            fabric_hints_->ep_attr->type = FI_EP_RDM;
#endif

            // by default, we will always want completions on both tx/rx events
            fabric_hints_->tx_attr->op_flags = FI_COMPLETION;
            fabric_hints_->rx_attr->op_flags = FI_COMPLETION;

            uint64_t flags = 0;
            cnt_deb.debug(hpx::debug::str<>("get fabric info"));
            int ret = fi_getinfo(FI_VERSION(GHEX_LIBFABRIC_FI_VERSION_MAJOR, GHEX_LIBFABRIC_FI_VERSION_MINOR),
                nullptr, nullptr, flags, fabric_hints_, &fabric_info_);
            if (ret) {
                throw fabric_error(ret, "Failed to get fabric info");
            }
            cnt_deb.trace(hpx::debug::str<>("Fabric info"), "\n"
                , fi_tostr(fabric_info_, FI_TYPE_INFO));

            immediate_ = (fabric_info_->rx_attr->mode & FI_RX_CQ_DATA)!=0;
            cnt_deb.trace(hpx::debug::str<>("Fabric IMMEDIATE")
                , immediate_);
//            bool context = cnt_deb.declare_variable<bool>(
//                        fabric_hints_->mode & FI_CONTEXT);
            bool context = (fabric_hints_->mode & FI_CONTEXT)!=0;
            cnt_deb.debug(hpx::debug::str<>("Requires FI_CONTEXT")
                , context);

            cnt_deb.debug(hpx::debug::str<>("Creating fi_fabric"));
            ret = fi_fabric(fabric_info_->fabric_attr, &fabric_, nullptr);
            if (ret) {
                throw fabric_error(ret, "Failed to get fi_fabric");
            }

            // Allocate a domain.
            cnt_deb.debug(hpx::debug::str<>("Allocating domain"));
            ret = fi_domain(fabric_, fabric_info_, &fabric_domain_, nullptr);
            if (ret) throw fabric_error(ret, "fi_domain");

            // Cray specific. Disable memory registration cache
            _set_disable_registration();

            fi_freeinfo(fabric_hints_);
            FUNC_END_DEBUG_MSG;
        }

        // -------------------------------------------------------------------
        // create endpoint and get ready for possible communications
        void startup()
        {
            FUNC_START_DEBUG_MSG;
            // only allow one thread to startup the controller
            static std::atomic_flag initialized = ATOMIC_FLAG_INIT;
            if (initialized.test_and_set()) return;
#ifdef GHEX_LIBFABRIC_ENDPOINT_RDM
            bind_endpoint_to_queues(ep_active_);
#endif

            // filling our vector of receivers...
            std::size_t num_receivers = GHEX_LIBFABRIC_MAX_PREPOSTS;
            receivers_.reserve(num_receivers);
            for(std::size_t i = 0; i != num_receivers; ++i)
            {
                receivers_.emplace_back(ep_active_, *memory_pool_);
            }


            for (std::size_t i = 0; i < GHEX_LIBFABRIC_MAX_SENDS; ++i)
            {
                sender *snd =
                   new sender(this,
                        ep_active_,
                        get_domain(),
                        memory_pool_.get());
                // after a sender has been used, it's postprocess handler
                // is called, this returns it to the free list
                snd->postprocess_handler_ = [this](sender* s)
                    {
                        --senders_in_use_;
                        cnt_deb.debug(hpx::debug::str<>("senders in use")
                                      , "(-- stack sender)"
                                      , hpx::debug::ptr(s)
                                      , hpx::debug::dec<>(senders_in_use_));
                        senders_.push(s);
//                        trigger_pending_work();
                    };
                // put the new sender on the free list
                senders_.push(snd);
            }

            FUNC_END_DEBUG_MSG;
        }

        // --------------------------------------------------------------------
        // return a sender object
        // --------------------------------------------------------------------
        sender* get_sender(libfabric::locality const& dest)
        {
            sender* snd = nullptr;
            if (senders_.pop(snd))
            {
                snd->dst_addr_ = dest.fi_address();
                ++senders_in_use_;
                cnt_deb.debug(hpx::debug::str<>("get_sender")
                    , hpx::debug::ptr(snd)
                    , "get address from", iplocality(here_)
                    , "to" , iplocality(dest)
                    , "fi_addr (rank)" , hpx::debug::hex<4>(snd->dst_addr_)
                    , "senders in use (++ get_sender)"
                    , hpx::debug::dec<>(senders_in_use_));
            }
            return snd;
        }

        // --------------------------------------------------------------------
        // return a sender object
        // --------------------------------------------------------------------
        sender* get_sender(int dest_rank)
        {
            sender* snd = nullptr;
            if (senders_.pop(snd))
            {
                // @TODO - fix this to only use rank = fi_addr_t(dest_rank)
                libfabric::locality dest;
                std::size_t addrlen = libfabric::locality_defs::array_size;
                int ret = fi_av_lookup(av_, fi_addr_t(dest_rank),
                                       dest.fabric_data_writable(), &addrlen);

                snd->dst_addr_ = dest.fi_address();
                ++senders_in_use_;
                cnt_deb.debug(hpx::debug::str<>("get_sender")
                    , hpx::debug::ptr(snd)
                    , "get address from", iplocality(here_)
                    , "to" , iplocality(dest)
                    , "fi_addr (rank)" , hpx::debug::hex<4>(snd->dst_addr_)
                    , "senders in use (++ get_sender)"
                    , hpx::debug::dec<>(senders_in_use_));
            }
            return snd;
        }

        // --------------------------------------------------------------------
        // Special GNI extensions to disable memory registration cache

        // this helper function only works for string ops
        void _set_check_domain_op_value(int op, const char *value)
        {
#ifdef GHEX_LIBFABRIC_PROVIDER_GNI
            int ret;
            struct fi_gni_ops_domain *gni_domain_ops;
            char *get_val;

            ret = fi_open_ops(&fabric_domain_->fid, FI_GNI_DOMAIN_OPS_1,
                      0, (void **) &gni_domain_ops, nullptr);
            if (ret) throw fabric_error(ret, "fi_open_ops");
            cnt_deb.debug("domain ops returned " , hpx::debug::ptr(gni_domain_ops));

            ret = gni_domain_ops->set_val(&fabric_domain_->fid,
                    (dom_ops_val_t)(op), &value);
            if (ret) throw fabric_error(ret, "set val (ops)");

            ret = gni_domain_ops->get_val(&fabric_domain_->fid,
                    (dom_ops_val_t)(op), &get_val);
            cnt_deb.debug("Cache mode set to " , get_val);
            if (std::string(value) != std::string(get_val))
                throw fabric_error(ret, "get val");
#endif
        }

        void _set_disable_registration()
        {
#ifdef GHEX_LIBFABRIC_PROVIDER_GNI
            _set_check_domain_op_value(GNI_MR_CACHE, "none");
#endif
        }

        // --------------------------------------------------------------------
        locality create_local_endpoint()
        {
            struct fid *id;
            int ret;
#ifdef GHEX_LIBFABRIC_ENDPOINT_RDM
            cnt_deb.debug(hpx::debug::str<>("active endpoint"));
            new_endpoint_active(fabric_info_, &ep_active_);
            cnt_deb.debug(hpx::debug::str<>("active endpoint") , hpx::debug::ptr(ep_active_));
            id = &ep_active_->fid;
#endif

#if defined(GHEX_LIBFABRIC_PROVIDER_SOCKETS)
            // with tcp we do not use PMI boot, so enable the endpoint now
            cnt_deb.debug(hpx::debug::str<>("Enabling (SOCKETS)") , hpx::debug::ptr(ep_active_));
            ret = fi_enable(ep_active_);
            if (ret) throw fabric_error(ret, "fi_enable");
#endif

            locality::locality_data local_addr;
            std::size_t addrlen = locality_defs::array_size;
            cnt_deb.debug(hpx::debug::str<>("Get address : size") , hpx::debug::dec<>(addrlen));
            ret = fi_getname(id, local_addr.data(), &addrlen);
            if (ret || (addrlen>locality_defs::array_size)) {
                fabric_error(ret, "fi_getname - size error or other problem");
            }

            // optimized out when debug logging is false
            if (cnt_deb.is_enabled())
            {
                std::stringstream temp1;
                for (std::size_t i=0; i<locality_defs::array_length; ++i) {
                    temp1 << hpx::debug::ipaddr(&local_addr[i]) << " - ";
                }
                cnt_deb.debug(hpx::debug::str<>("raw address data") , temp1.str().c_str());
                std::stringstream temp2;
                for (std::size_t i=0; i<locality_defs::array_length; ++i) {
                    temp2 << hpx::debug::hex<8>(local_addr[i]) << " - ";
                }
                cnt_deb.debug(hpx::debug::str<>("raw address data") , temp2.str().c_str());
            };
            FUNC_END_DEBUG_MSG;
            return locality(local_addr);
        }

        // --------------------------------------------------------------------
        void new_endpoint_active(struct fi_info *info, struct fid_ep **new_endpoint)
        {
            FUNC_START_DEBUG_MSG;
            // create an 'active' endpoint that can be used for sending/receiving
            cnt_deb.debug(hpx::debug::str<>("Active endpoint"));
            cnt_deb.debug(hpx::debug::str<>("Got info mode") , (info->mode & FI_NOTIFY_FLAGS_ONLY));
            int ret = fi_endpoint(fabric_domain_, info, new_endpoint, nullptr);
            if (ret) throw fabric_error(ret, "fi_endpoint");

            if (info->ep_attr->type == FI_EP_MSG) {
                if (event_queue_) {
                    cnt_deb.debug(hpx::debug::str<>("Binding endpoint to EQ"));
                    ret = fi_ep_bind(*new_endpoint, &event_queue_->fid, 0);
                    if (ret) throw fabric_error(ret, "bind event_queue_");
                }
            }
        }

        // --------------------------------------------------------------------
        void bind_endpoint_to_queues(struct fid_ep *endpoint)
        {
            int ret;
            if (av_) {
                cnt_deb.debug(hpx::debug::str<>("Binding AV"));
                ret = fi_ep_bind(endpoint, &av_->fid, 0);
                if (ret) throw fabric_error(ret, "bind event_queue_");
            }

            if (txcq_) {
                cnt_deb.debug(hpx::debug::str<>("Binding TX CQ"));
                ret = fi_ep_bind(endpoint, &txcq_->fid, FI_TRANSMIT);
                if (ret) throw fabric_error(ret, "bind txcq");
            }

            if (rxcq_) {
                cnt_deb.debug(hpx::debug::str<>("Binding RX CQ"));
                ret = fi_ep_bind(endpoint, &rxcq_->fid, FI_RECV);
                if (ret) throw fabric_error(ret, "rxcq");
            }

            if (ep_shared_rx_cxt_) {
                cnt_deb.debug(hpx::debug::str<>("Binding RX context"));
                ret = fi_ep_bind(endpoint, &ep_shared_rx_cxt_->fid, 0);
                if (ret) throw fabric_error(ret, "ep_shared_rx_cxt_");
            }

            cnt_deb.debug(hpx::debug::str<>("Enabling endpoint")
                , hpx::debug::ptr(endpoint));
            ret = fi_enable(endpoint);
            if (ret) throw fabric_error(ret, "fi_enable");

            FUNC_END_DEBUG_MSG;
        }

        // --------------------------------------------------------------------
        // if we did not bootstrap, then fetch the list of all localities
        // from agas and insert each one into the address vector
        void initialize_localities()
        {
            FUNC_START_DEBUG_MSG;
#ifndef GHEX_LIBFABRIC_HAVE_BOOTSTRAPPING
            cnt_deb.debug(hpx::debug::str<>("initialize_localities")
                , m_size_ , " localities");

            // make sure address vector is created
            create_completion_queues(fabric_info_, m_size_ );

            MPI_exchange_localities();

#endif
            cnt_deb.debug(hpx::debug::str<>("Done localities"));
            FUNC_END_DEBUG_MSG;
        }

        // --------------------------------------------------------------------
        const locality & here() const { return here_; }

        // --------------------------------------------------------------------
        const bool & immedate_data_supported() const { return immediate_; }

        // --------------------------------------------------------------------
        // returns true when all connections have been disconnected and none are active
        bool isTerminated() {
            return false;
            //return (qp_endpoint_map_.size() == 0);
        }

        // types we need for connection and disconnection callback functions
        // into the main parcelport code.
        typedef std::function<void(fid_ep *endpoint, uint32_t ipaddr)>
            ConnectionFunction;
        typedef std::function<void(fid_ep *endpoint, uint32_t ipaddr)>
            DisconnectionFunction;

//        typedef std::function<void(libfabric::controller *controller,
//                                   const libfabric::locality &remote_addr)>
//            BootstrapFunction;

        // --------------------------------------------------------------------

        // send full address list back to the address that contacted us
        void update_bootstrap_connections()
        {
            cnt_deb.debug(hpx::debug::str<>("accepting bootstrap"));
            if (--bootstrap_counter_ == 0) {
                cnt_deb.debug(hpx::debug::str<>("bootstrap clients"));
                std::size_t N = get_num_localities();
                //
                std::vector<libfabric::locality> addresses;
                addresses.reserve(N);
                //
                libfabric::locality addr;
                std::size_t addrlen = libfabric::locality_defs::array_size;
                for (std::size_t i=0; i<N; ++i) {
                    int ret = fi_av_lookup(av_, fi_addr_t(i),
                                           addr.fabric_data_writable(), &addrlen);
                    if ((ret == 0) && (addrlen==libfabric::locality_defs::array_size)) {
                        addr.set_fi_address(fi_addr_t(i));
                        cnt_deb.debug(hpx::debug::str<>("bootstrap sending") , iplocality(addr));
                        addresses.push_back(addr);
                    }
                    else {
                        throw std::runtime_error("address vector traversal failure");
                    }
                }

                // don't send addresses to self, start at index=1
                for (std::size_t i=1; i<N; ++i) {
                    cnt_deb.debug(hpx::debug::str<>("Sending bootstrap list")
                        , iplocality(addresses[i]));

//                    parcelport_->send_raw_data(addresses[i]
//                        , addresses.data()
//                        , N*sizeof(libfabric::locality)
//                        , libfabric::header<
//                            GHEX_LIBFABRIC_MESSAGE_HEADER_SIZE>::bootstrap_flag);

                }
                //parcelport_->set_bootstrap_complete();
            }
        }

        // --------------------------------------------------------------------
        // This is the main polling function that checks for work completions
        // and connection manager events, if stopped is true, then completions
        // are thrown away, otherwise the completion callback is triggered
        unsigned int poll_endpoints(bool /*stopped*/=false)
        {
            unsigned int work = poll_for_work_completions();

#ifdef GHEX_LIBFABRIC_ENDPOINT_MSG
            work += poll_event_queue(stopped);
#endif
            return work;
        }

        // --------------------------------------------------------------------
        unsigned int poll_for_work_completions()
        {
            // @TODO, disable polling until queues are initialized to avoid this check
            // if queues are not setup, don't poll
            if (HPX_UNLIKELY(!rxcq_)) return 0;
            //
            return poll_send_queue() + poll_recv_queue();
        }

        // --------------------------------------------------------------------
        unsigned int poll_send_queue()
        {
//            LOG_TIMED_INIT(poll);
//            LOG_TIMED_BLOCK(poll, DEVEL, 5.0, { cnt_deb.debug("poll_send_queue"); });

            fi_cq_msg_entry entry;
            int ret = fi_cq_read(txcq_, &entry, 1);
            //
            if (ret>0) {
                cnt_deb.debug(hpx::debug::str<>("Completion")
                    , "txcq wr_id"
                    , fi_tostr(&entry.flags, FI_TYPE_OP_FLAGS)
                    , "(" , hpx::debug::dec<>(entry.flags) , ")"
                    , "context" , hpx::debug::ptr(entry.op_context)
                    , "length" , hpx::debug::hex<8>(entry.len));
                if (entry.flags & FI_RMA) {
                    cnt_deb.debug(hpx::debug::str<>("Completion")
                        , "txcq RMA"
                        , "Context " , hpx::debug::ptr(entry.op_context));
                    rma_receiver* rcv = reinterpret_cast<rma_receiver*>(entry.op_context);
                    rcv->handle_rma_read_completion();
                }
                else if (entry.flags == (FI_MSG | FI_SEND)) {
                    cnt_deb.debug(hpx::debug::str<>("Completion")
                        , "txcq MSG send completion");
                    sender* handler = reinterpret_cast<sender*>(entry.op_context);
                    handler->handle_send_completion();
                }
                else {
                    cnt_deb.error("$$$$$ Received an unknown txcq completion *****"
                        , hpx::debug::dec<>(entry.flags));
                    std::terminate();
                }
                return 1;
            }
            else if (ret==0 || ret==-FI_EAGAIN) {
                // do nothing, we will try again on the next check
//                LOG_TIMED_MSG(poll, DEVEL, 10, "txcq " , (ret==0?"---":"FI_EAGAIN"));
            }
            else if (ret == -FI_EAVAIL) {
                struct fi_cq_err_entry e = {};
                int err_sz = fi_cq_readerr(txcq_, &e ,0);
                HPX_UNUSED(err_sz);
                // from the manpage 'man 3 fi_cq_readerr'
                // On error, a negative value corresponding to
                // 'fabric errno' is returned
                if(e.err == err_sz) {
                    cnt_deb.error(
                          "txcq Error FI_EAVAIL with len " , hpx::debug::hex<6>(e.len)
                        , "context " , hpx::debug::ptr(e.op_context));
                }
                // flags might not be set correctly
                if (e.flags == (FI_MSG | FI_SEND)) {
                    cnt_deb.error("txcq Error FI_EAVAIL for FI_SEND with len " , hpx::debug::hex<6>(e.len)
                        , "context " , hpx::debug::ptr(e.op_context));
                }
                if (e.flags & FI_RMA) {
                    cnt_deb.error("txcq Error FI_EAVAIL for FI_RMA with len" , hpx::debug::hex<6>(e.len)
                        , "context " , hpx::debug::ptr(e.op_context));
                }
                rma_base *base = reinterpret_cast<rma_base*>(e.op_context);
                switch (base->context_type()) {
                    case ctx_sender:
                        reinterpret_cast<sender*>(e.op_context)->handle_error(e);
                        break;
                    case ctx_receiver:
                        reinterpret_cast<receiver*>(e.op_context)->handle_error(e);
                        break;
                    case ctx_rma_receiver:
                        reinterpret_cast<rma_receiver*>(e.op_context)->handle_error(e);
                        break;
                }
            }
            else {
                cnt_deb.error("unknown error in completion txcq read");
            }
            return 0;
        }

        // --------------------------------------------------------------------
        unsigned int poll_recv_queue()
        {
//            LOG_TIMED_INIT(poll);
//            LOG_TIMED_BLOCK(poll, DEVEL, 5.0, { cnt_deb.debug("poll_recv_queue"); });

            int result = 0;
            fi_addr_t src_addr;
            fi_cq_msg_entry entry;

            // receives will use fi_cq_readfrom as we want the source address
            int ret = fi_cq_readfrom(rxcq_, &entry, 1, &src_addr);
            //
            if (ret>0) {
                cnt_deb.debug(hpx::debug::str<>("Completion")
                    , "rxcq wr_id "
                    , fi_tostr(&entry.flags, FI_TYPE_OP_FLAGS)
                    , "(" , hpx::debug::dec<>(entry.flags) , ")"
                    , "source" , hpx::debug::ptr(src_addr)
                    , "context" , hpx::debug::ptr(entry.op_context)
                    , "length" , hpx::debug::hex<8>(entry.len));
                if (src_addr == FI_ADDR_NOTAVAIL)
                {
                    cnt_deb.debug(hpx::debug::str<>("New connection?")
                        , "(bootstrap): "
                        , "Source address not available...");
                    reinterpret_cast<receiver *>(entry.op_context)->
                        handle_new_connection(this, entry.len);
                }

                if ((entry.flags & FI_RMA) == FI_RMA) {
                    cnt_deb.debug(hpx::debug::str<>("Completion")
                        , "rxcq RMA");
                }

                else if (entry.flags == (FI_MSG | FI_RECV)) {
                    cnt_deb.debug(hpx::debug::str<>("Completion")
                        , "rxcq recv completion "
                        , hpx::debug::ptr(entry.op_context));
                    reinterpret_cast<receiver *>(entry.op_context)->
                        handle_recv(src_addr, entry.len);
                }
                else {
                    cnt_deb.error("Received an unknown rxcq completion "
                        , hpx::debug::dec<>(entry.flags));
                    std::terminate();
                }
                result = 1;
            }
            else if (ret==0 || ret==-FI_EAGAIN) {
                // do nothing, we will try again on the next check
//                LOG_TIMED_MSG(poll, DEVEL, 10, "rxcq " , (ret==0?"---":"FI_EAGAIN"));
            }
            else if (ret == -FI_EAVAIL) {
                // read the full error status
                struct fi_cq_err_entry e = {};
                int err_sz = fi_cq_readerr(rxcq_, &e ,0);
                HPX_UNUSED(err_sz);
                // from the manpage 'man 3 fi_cq_readerr'
                //
                cnt_deb.error("rxcq Error ??? "
                              , "err "     , hpx::debug::dec<>(-e.err)
                              , "flags "   , hpx::debug::hex<6>(e.flags)
                              , "len "     , hpx::debug::hex<6>(e.len)
                              , "context " , hpx::debug::ptr(e.op_context)
                              , "error "   ,
                fi_cq_strerror(rxcq_, e.prov_errno, e.err_data, (char*)e.buf, e.len));
                std::terminate();
            }
            else {
                cnt_deb.error("unknown error in completion rxcq read");
            }
            return result;
        }

        // --------------------------------------------------------------------
        inline struct fid_domain * get_domain() {
            return fabric_domain_;
        }

        // --------------------------------------------------------------------
        inline rma::memory_pool<libfabric_region_provider>& get_memory_pool() {
            return *memory_pool_;
        }

        // --------------------------------------------------------------------
        void create_completion_queues(struct fi_info *info, int N)
        {
            FUNC_START_DEBUG_MSG;

            // only one thread must be allowed to create queues,
            // and it is only required once
            scoped_lock lock(initialization_mutex_);
            if (txcq_!=nullptr || rxcq_!=nullptr || av_!=nullptr) {
                return;
            }

            int ret;

            fi_cq_attr cq_attr = {};
            // @TODO - why do we check this
//             if (cq_attr.format == FI_CQ_FORMAT_UNSPEC) {
                cnt_deb.debug(hpx::debug::str<>("Setting CQ")
                    , "FI_CQ_FORMAT_MSG");
                cq_attr.format = FI_CQ_FORMAT_MSG;
//             }

            // open completion queue on fabric domain and set context ptr to tx queue
            cq_attr.wait_obj = FI_WAIT_NONE;
            cq_attr.size = info->tx_attr->size;
            info->tx_attr->op_flags |= FI_COMPLETION;
            cq_attr.flags = 0;//|= FI_COMPLETION;
            cnt_deb.debug(hpx::debug::str<>("Creating tx CQ")
                          , "size" , hpx::debug::dec<>(info->tx_attr->size));
            ret = fi_cq_open(fabric_domain_, &cq_attr, &txcq_, &txcq_);
            if (ret) throw fabric_error(ret, "fi_cq_open");

            // open completion queue on fabric domain and set context ptr to rx queue
            cq_attr.size = info->rx_attr->size;
            cnt_deb.debug(hpx::debug::str<>("Creating rx CQ")
                          , "size" , hpx::debug::dec<>(info->rx_attr->size));
            ret = fi_cq_open(fabric_domain_, &cq_attr, &rxcq_, &rxcq_);
            if (ret) throw fabric_error(ret, "fi_cq_open");


            fi_av_attr av_attr = {};
            if (info->ep_attr->type == FI_EP_RDM || info->ep_attr->type == FI_EP_DGRAM) {
                if (info->domain_attr->av_type != FI_AV_UNSPEC)
                    av_attr.type = info->domain_attr->av_type;
                else {
                    cnt_deb.debug(hpx::debug::str<>("map FI_AV_TABLE"));
                    av_attr.type  = FI_AV_TABLE;
                    av_attr.count = N;
                }

                cnt_deb.debug(hpx::debug::str<>("Creating AV"));
                ret = fi_av_open(fabric_domain_, &av_attr, &av_, nullptr);
                if (ret) throw fabric_error(ret, "fi_av_open");
            }
            FUNC_END_DEBUG_MSG;
        }

        // --------------------------------------------------------------------
        // needed at bootstrap time to find the correct fi_addr_t for
        // a locality
        bool resolve_address(libfabric::locality &address)
        {
            std::size_t N = get_num_localities();
            std::size_t addrlen = libfabric::locality_defs::array_size;
            libfabric::locality addr;
            for (std::size_t i=0; i<N; ++i) {
                int ret = fi_av_lookup(av_, fi_addr_t(i),
                                       addr.fabric_data_writable(), &addrlen);
                if ((ret == 0) && (addrlen==libfabric::locality_defs::array_size)) {
                    if (addr == address) {
                        address.set_fi_address(fi_addr_t(i));
                        return true;
                    }
                }
                else {
                    throw std::runtime_error(
                        "address vector resolve_address failure");
                }
            }
            return false;
        }

        // --------------------------------------------------------------------
        libfabric::locality insert_address(const libfabric::locality &address)
        {
            FUNC_START_DEBUG_MSG;
            cnt_deb.trace(hpx::debug::str<>("inserting AV"), iplocality(address));
            fi_addr_t fi_addr = 0xffffffff;
            int ret = fi_av_insert(av_, address.fabric_data(), 1, &fi_addr, 0, nullptr);
            if (ret < 0) {
                fabric_error(ret, "fi_av_insert");
            }
            else if (ret == 0) {
                cnt_deb.error("fi_av_insert called with existing address");
                fabric_error(ret, "fi_av_insert did not return 1");
            }
            // address was generated correctly, now update the locality with the fi_addr
            libfabric::locality new_locality(address, fi_addr);
            cnt_deb.trace(hpx::debug::str<>("AV add")
                          , "rank" , hpx::debug::dec<>(fi_addr)
                          , iplocality(new_locality)
                          , "fi_addr " , hpx::debug::hex<4>(fi_addr));
            FUNC_END_DEBUG_MSG;
            return new_locality;
        }

    private:
        // store info about local device
        std::string  device_;
        std::string  interface_;
        sockaddr_in  local_addr_;

        // Pinned memory pool used for allocating buffers
        std::unique_ptr<rma::memory_pool<libfabric_region_provider>> memory_pool_;

        // Shared completion queue for all endoints
        // Count outstanding receives posted to SRQ + Completion queue
        std::vector<receiver> receivers_;

        // only allow one thread to handle connect/disconnect events etc
        mutex_type            initialization_mutex_;
    };

}}}

#endif
