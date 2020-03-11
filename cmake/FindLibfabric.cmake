find_package(PkgConfig QUIET)
pkg_check_modules(PC_LIBFABRIC QUIET libfabric)

find_path(LIBFABRIC_INCLUDE_DIR rdma/fabric.h
  HINTS
    ${LIBFABRIC_ROOT} ENV LIBFABRIC_ROOT
    ${LIBFABRIC_DIR} ENV LIBFABRIC_DIR
  PATH_SUFFIXES include)

find_library(LIBFABRIC_LIBRARY NAMES fabric
  HINTS
    ${LIBFABRIC_ROOT} ENV LIBFABRIC_ROOT
  PATH_SUFFIXES lib lib64)

set(LIBFABRIC_LIBRARIES    ${LIBFABRIC_LIBRARY} CACHE INTERNAL "")
set(LIBFABRIC_INCLUDE_DIRS ${LIBFABRIC_INCLUDE_DIR} CACHE INTERNAL "")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Libfabric DEFAULT_MSG
  LIBFABRIC_LIBRARY LIBFABRIC_INCLUDE_DIR)

mark_as_advanced(LIBFABRIC_ROOT)

if(NOT TARGET libfabric::libfabric AND Libfabric_FOUND)
  add_library(libfabric::libfabric SHARED IMPORTED)
  set_target_properties(libfabric::libfabric PROPERTIES
    IMPORTED_LOCATION ${LIBFABRIC_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${LIBFABRIC_INCLUDE_DIR}
  )
endif()

#  target_link_libraries(hpx::libfabric INTERFACE ${LIBFABRIC_LIBRARY})
