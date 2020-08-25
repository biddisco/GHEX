
set(_tests test_low_level test_send_multi test_barrier test_barrier_mt test_cancel test_context test_send_recv)

foreach(t_ ${_tests})
    add_executable( ${t_} ./${t_}.cpp )
    target_link_libraries(${t_} gtest_main_mt)
    add_test(
        NAME ${t_}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${t_}> ${MPIEXEC_POSTFLAGS}
    )

    if (GHEX_USE_UCP)
        add_executable( ${t_}_ucx ./${t_}.cpp )
        target_link_libraries(${t_}_ucx gtest_main_ucx)
        target_compile_definitions(${t_}_ucx PUBLIC GHEX_TEST_USE_UCX)
        if (GHEX_USE_PMIX)
            target_compile_definitions(${t_}_ucx PRIVATE GHEX_USE_PMI)
        endif()
        add_test(
            NAME ${t_}_ucx
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${t_}_ucx> ${MPIEXEC_POSTFLAGS}
        )
    endif()

    if (GHEX_USE_LIBFABRIC)
        add_executable( ${t_}_libfabric ./${t_}.cpp )
        target_link_libraries(${t_}_libfabric gtest_main_libfabric)
        target_compile_definitions(${t_}_libfabric PUBLIC GHEX_TEST_USE_LIBFABRIC)
        if (GHEX_USE_PMIX)
            target_compile_definitions(${t_}_libfabric PRIVATE GHEX_USE_PMI)
        endif()
        add_test(
            NAME ${t_}_libfabric
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${t_}_libfabric> ${MPIEXEC_POSTFLAGS}
        )
    endif()

endforeach(t_ ${_tests})

if (GHEX_USE_LIBFABRIC)
    add_executable( test_fancy_pointer test_fancy_pointer.cpp )
    target_link_libraries(test_fancy_pointer gtest_main_libfabric)
    target_compile_definitions(test_fancy_pointer PUBLIC GHEX_TEST_USE_LIBFABRIC)
endif()

if (GHEX_USE_UCP)
    set(_tests_ucx test_ucx_context)

    foreach(t_ ${_tests_ucx})

        add_executable( ${t_} ./${t_}.cpp )
        target_link_libraries(${t_} gtest_main_ucx)
        if (GHEX_USE_PMIX)
            target_compile_definitions(${t_} PRIVATE GHEX_USE_PMI)
        endif()
        add_test(
            NAME ${t_}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${t_}> ${MPIEXEC_POSTFLAGS}
        )

    endforeach(t_ ${_tests_ucx})
endif()
