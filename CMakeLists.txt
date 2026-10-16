cmake_minimum_required(VERSION 3.24...3.27)
project(
    LIBHERIC
    VERSION 
        0.1
    DESCRIPTION 
        "Library for interfacing with several ERI acceleration libraries targeting multiple hardware platforms"
    LANGUAGES 
        C CXX Fortran
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FindPkgConfig) 
include(CheckCXXSourceCompiles)

#add_subdirectory(libcint_rtwt)

#find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(libint2 REQUIRED)
find_package(vectorclass REQUIRED)
find_package(HiGHS REQUIRED)
#pkg_search_module(hwloc REQUIRED IMPORTED_TARGET hwloc)



if (NOT DEFINED MAX_AM)
    set(MAX_AM 3)
endif()

if (NOT DEFINED FSTR_LEN)
    set(FSTR_LEN 256)
endif()

if (NOT DEFINED MAX_DEVICES_ON_NODE)
    set(MAX_DEVICES_ON_NODE 32)
endif()

if (NOT DEFINED MAX_SET_SIZE)
    set(MAX_SET_SIZE 64)
endif()

if (NOT DEFINED LOG_PRINT)
    set(LOG_PRINT false)
else()
    set(LOG_PRINT true)
endif()

if (NOT DEFINED DEBUG_PRINT)
    set(DEBUG_PRINT false)
else()
    set(DEBUG_PRINT true)
    set(LOG_PRINT true)
endif()

configure_file(
    include/constants.h.in
    include/constants.h
)
configure_file(
    include/fortran_interface.f08.in
    include/fortran_interface.f08
)

add_library(
    libheric 
    SHARED 
        src/libheric.cpp include/libheric.h
        src/logging.cpp include/logging.h
        src/abstract_acc.cpp include/abstract_acc.h 
        src/cpu_acc.cpp include/cpu_acc.h 
        src/cuda_acc.cpp include/cuda_acc.h
        src/hip_acc.cpp include/hip_acc.h
        src/oneapi_acc.cpp include/oneapi_acc.h
        src/xrt_acc.cpp include/xrt_acc.h
        src/quartet_input.cpp include/quartet_input.h
        src/quartet_queue.cpp include/quartet_queue.h
        src/transform.cpp include/transform.h
        src/simulation_box.cpp include/simulation_box.h
        src/util.cpp include/util.h
        src/rys_reference.cpp include/rys_reference.h
        src/optimize.cpp include/optimize.h
        include/fortran_interface.f08
)
target_compile_options(libheric PRIVATE
    $<$<CONFIG:Debug>:-Wall;-Wextra>
)
target_include_directories(
    libheric 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include> 
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/include
)

set(LIBHERIC_FLAGS "-DEIGEN_DONT_PARALLELIZE")
set(LIBHERIC_PUBLIC_LIBS "")
set(LIBHERIC_PRIVATE_LIBS "")
list(APPEND LIBHERIC_PUBLIC_LIBS OpenMP::OpenMP_CXX)
list(APPEND LIBHERIC_PRIVATE_LIBS highs::highs)
list(APPEND LIBHERIC_PRIVATE_LIBS Eigen3::Eigen)
list(APPEND LIBHERIC_PRIVATE_LIBS vectorclass)
list(APPEND LIBHERIC_PRIVATE_LIBS Libint2::int-static)
#list(APPEND LIBHERIC_PRIVATE_LIBS cintroots )

if (ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND LIBHERIC_FLAGS "-DENABLE_CUDA")
    list(APPEND LIBHERIC_PUBLIC_LIBS CUDA::cudart)
endif()

if (ENABLE_HIP)
    find_package(HIP REQUIRED)
    list(APPEND LIBHERIC_FLAGS "-DENABLE_HIP")
    list(APPEND LIBHERIC_PUBLIC_LIBS  hip::host)
endif()

if (ENABLE_ONEAPI)
    find_package(IntelSYCLForGCC REQUIRED)
    list(APPEND LIBHERIC_FLAGS "-DENABLE_ONEAPI")
    list(APPEND LIBHERIC_PUBLIC_LIBS  IntelSYCLForGCC)
endif()

if (ENABLE_XRT)
    find_package(XRT REQUIRED)
    list(APPEND LIBHERIC_FLAGS "-DENABLE_XRT")
    list(APPEND LIBHERIC_PUBLIC_LIBS  XRT::xrt_coreutil)
    target_include_directories(heric SYSTEM PRIVATE ${XRT_INCLUDE_DIRS})
endif()

target_link_libraries(libheric PUBLIC ${LIBHERIC_PUBLIC_LIBS})
target_link_libraries(libheric PRIVATE ${LIBHERIC_PRIVATE_LIBS})

set_property(TARGET libheric PROPERTY CXX_STANDARD 20)
target_compile_options(libheric PUBLIC -march=native)
target_compile_options(libheric PRIVATE ${LIBHERIC_FLAGS})

add_executable(libint_test tests/libint_test.cpp)
set_property(TARGET libint_test PROPERTY CXX_STANDARD 20)
get_target_property(_LIBINT_INCL Libint2::int-static INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(
    libint_test 
    PRIVATE 
        "${CMAKE_CURRENT_BINARY_DIR}/include"
        "${_LIBINT_INCL}"
        "${_LIBCOMPRESS_INCL}"
)
target_link_libraries(libint_test PUBLIC libheric vectorclass Eigen3::Eigen)

add_executable(queue_test tests/queue_test.cpp)
set_property(TARGET queue_test PROPERTY CXX_STANDARD 20)
target_include_directories(
    queue_test 
    PRIVATE 
        "${CMAKE_CURRENT_BINARY_DIR}/include"
        "${_LIBINT_INCL}"
        "${_LIBCOMPRESS_INCL}"
)
target_link_libraries(queue_test PUBLIC libheric vectorclass Eigen3::Eigen)

add_executable(opt_screening_test tests/opt_screening_test.cpp)
target_include_directories(
    opt_screening_test 
    PRIVATE 
        "${CMAKE_CURRENT_BINARY_DIR}/include"
)
target_link_libraries(opt_screening_test PUBLIC libheric)

#add_executable(fortran_interface_test tests/fortran_interface_test.f08)
#target_include_directories(
#    fortran_interface_test 
#    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include"
#)
#target_link_libraries(fortran_interface_test PUBLIC libheric MPI::MPI_Fortran OpenMP::OpenMP_Fortran)

#add_executable(hwloc_test tests/hwloc_test.cpp)
#target_link_libraries(hwloc_test PRIVATE PkgConfig::hwloc)
#
#add_executable(hwloc_mpi tests/hwloc_mpi.cpp)
#target_link_libraries(
#    hwloc_mpi 
#    PRIVATE
#        libheric 
#        PkgConfig::hwloc
#        MPI::MPI_C
#)

install(
    TARGETS libheric 
    EXPORT  libheric_targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES 
            ${CMAKE_CURRENT_BINARY_DIR}/libheric_interface.mod
        DESTINATION 
            ${CMAKE_INSTALL_INCLUDEDIR}) 

install(
    EXPORT libheric_targets
    FILE libheric_targets.cmake
    NAMESPACE libheric::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libheric
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/libheric-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libheric-config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/libheric"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libheric-config-version.cmake"
    COMPATIBILITY AnyNewerVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/libheric-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/libheric-config-version.cmake"
    DESTINATION
        "${CMAKE_INSTALL_LIBDIR}/cmake/libheric"
)

