include(GHEX_macros)

#------------------------------------------------------------------------------
# Logging
#------------------------------------------------------------------------------
ghex_option(GHEX_LIBFABRIC_WITH_LOGGING BOOL
  "Enable logging in the libfabric ParcelPort (default: OFF - Warning - severely impacts usability when enabled)"
  OFF CATEGORY "libfabric" ADVANCED)

if (GHEX_LIBFABRIC_WITH_LOGGING)
  ghex_add_config_define_namespace(
      DEFINE    GHEX_LIBFABRIC_HAVE_LOGGING
      NAMESPACE libfabric)
endif()

#------------------------------------------------------------------------------
# Bootstrap options
#------------------------------------------------------------------------------
ghex_option(GHEX_LIBFABRIC_WITH_BOOTSTRAPPING BOOL
  "Configure the transport layerto enable bootstrap capabilities"
  ${PMIx_FOUND} CATEGORY "libfabric" ADVANCED)

if (GHEX_LIBFABRIC_WITH_BOOTSTRAPPING)
  ghex_add_config_define_namespace(
      DEFINE    GHEX_LIBFABRIC_HAVE_BOOTSTRAPPING
      VALUE     std::true_type
      NAMESPACE libfabric)
endif()

#------------------------------------------------------------------------------
# Hardware device selection
#------------------------------------------------------------------------------
ghex_option(GHEX_LIBFABRIC_PROVIDER STRING
  "The provider (verbs/gni/psm2/sockets)"
  "verbs" CATEGORY "libfabric" ADVANCED)

ghex_add_config_define_namespace(
    DEFINE GHEX_LIBFABRIC_PROVIDER
    VALUE  "\"${GHEX_LIBFABRIC_PROVIDER}\""
    NAMESPACE libfabric)

if(GHEX_LIBFABRIC_PROVIDER MATCHES "verbs")
    ghex_add_config_define_namespace(
        DEFINE GHEX_LIBFABRIC_VERBS
        NAMESPACE libfabric)
elseif(GHEX_LIBFABRIC_PROVIDER MATCHES "gni")
    ghex_add_config_define_namespace(
        DEFINE GHEX_LIBFABRIC_GNI
        NAMESPACE libfabric)
    # enable bootstrapping, add pmi library
    set(GHEX_LIBFABRIC_WITH_BOOTSTRAPPING ON)
    set(_libfabric_libraries ${_libfabric_libraries} PMIx::libpmix)
elseif(GHEX_LIBFABRIC_PROVIDER MATCHES "sockets")
    ghex_add_config_define_namespace(
        DEFINE GHEX_LIBFABRIC_SOCKETS
        NAMESPACE libfabric)
    # enable bootstrapping
    set(GHEX_LIBFABRIC_WITH_BOOTSTRAPPING ON)
elseif(GHEX_LIBFABRIC_PROVIDER MATCHES "psm2")
    ghex_add_config_define_namespace(
        DEFINE GHEX_LIBFABRIC_PSM2
        NAMESPACE libfabric)
endif()

#------------------------------------------------------------------------------
# Domain and endpoint are fixed, but used to be options
# leaving options in case they are needed again to support other platforms
#------------------------------------------------------------------------------
ghex_option(GHEX_LIBFABRIC_DOMAIN STRING
  "The libfabric domain (leave blank for default"
  "" CATEGORY "libfabric" ADVANCED)

ghex_add_config_define_namespace(
    DEFINE GHEX_LIBFABRIC_DOMAIN
    VALUE  "\"${GHEX_LIBFABRIC_DOMAIN}\""
    NAMESPACE libfabric)

ghex_option(GHEX_LIBFABRIC_ENDPOINT STRING
  "The libfabric endpoint type (leave blank for default"
  "rdm" CATEGORY "libfabric" ADVANCED)

ghex_add_config_define_namespace(
    DEFINE GHEX_LIBFABRIC_ENDPOINT
    VALUE  "\"${GHEX_LIBFABRIC_ENDPOINT}\""
    NAMESPACE libfabric)

#------------------------------------------------------------------------------
# Performance counters
#------------------------------------------------------------------------------
ghex_option(GHEX_LIBFABRIC_WITH_PERFORMANCE_COUNTERS BOOL
  "Enable libfabric parcelport performance counters (default: OFF)"
  OFF CATEGORY "libfabric" ADVANCED)

if (GHEX_LIBFABRIC_WITH_PERFORMANCE_COUNTERS)
  ghex_add_config_define_namespace(
      DEFINE    GHEX_LIBFABRIC_HAVE_PERFORMANCE_COUNTERS
      NAMESPACE libfabric)
endif()

#------------------------------------------------------------------------------
# Memory chunk/reservation options
#------------------------------------------------------------------------------
ghex_option(GHEX_LIBFABRIC_MEMORY_CHUNK_SIZE STRING
  "Number of bytes a default chunk in the memory pool can hold (default: 4K)"
  "4096" CATEGORY "libfabric" ADVANCED)

ghex_option(GHEX_LIBFABRIC_64K_PAGES STRING
  "Number of 64K pages we reserve for default message buffers (default: 10)"
  "10" CATEGORY "libfabric" ADVANCED)

ghex_option(GHEX_LIBFABRIC_MEMORY_COPY_THRESHOLD STRING
  "Cutoff size over which data is never copied into existing buffers (default: 4K)"
  "4096" CATEGORY "libfabric" ADVANCED)

ghex_add_config_define_namespace(
    DEFINE    GHEX_LIBFABRIC_MEMORY_CHUNK_SIZE
    VALUE     ${GHEX_LIBFABRIC_MEMORY_CHUNK_SIZE}
    NAMESPACE libfabric)

# define the message header size to be equal to the chunk size
ghex_add_config_define_namespace(
    DEFINE    GHEX_LIBFABRIC_MESSAGE_HEADER_SIZE
    VALUE     ${GHEX_LIBFABRIC_MEMORY_CHUNK_SIZE}
    NAMESPACE libfabric)

ghex_add_config_define_namespace(
    DEFINE    GHEX_LIBFABRIC_64K_PAGES
    VALUE     ${GHEX_LIBFABRIC_64K_PAGES}
    NAMESPACE libfabric)

ghex_add_config_define_namespace(
    DEFINE    GHEX_LIBFABRIC_MEMORY_COPY_THRESHOLD
    VALUE     ${GHEX_LIBFABRIC_MEMORY_COPY_THRESHOLD}
    NAMESPACE libfabric)

#------------------------------------------------------------------------------
# Preposting options
#------------------------------------------------------------------------------
ghex_option(GHEX_LIBFABRIC_MAX_PREPOSTS STRING
  "The number of pre-posted receive buffers (default: 512)"
  "512" CATEGORY "libfabric" ADVANCED)

ghex_add_config_define_namespace(
    DEFINE    GHEX_LIBFABRIC_MAX_PREPOSTS
    VALUE     ${GHEX_LIBFABRIC_MAX_PREPOSTS}
    NAMESPACE libfabric)

#------------------------------------------------------------------------------
# Throttling options
#------------------------------------------------------------------------------
ghex_option(GHEX_LIBFABRIC_MAX_SENDS STRING
  "Threshold of active sends at which throttling is enabled (default: 16)"
  "16" CATEGORY "libfabric" ADVANCED)

ghex_add_config_define_namespace(
    DEFINE    GHEX_LIBFABRIC_MAX_SENDS
    VALUE     ${GHEX_LIBFABRIC_MAX_SENDS}
    NAMESPACE libfabric)

#------------------------------------------------------------------------------
# Write options to file in build dir
#------------------------------------------------------------------------------
write_config_defines_file(
    NAMESPACE libfabric
    FILENAME  "${CMAKE_BINARY_DIR}/ghex_libfabric_defines.hpp"
)
target_include_directories(libfabric::libfabric INTERFACE "${CMAKE_BINARY_DIR}")
