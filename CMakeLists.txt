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

set(CMAKE_CXX_FLAGS "-O0 -g -Wall -Wextra")
set(CMAKE_Fortran_FLAGS "-O0 -g -Wall -Wextra")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FindPkgConfig) 

find_package(CUDAToolkit REQUIRED)
find_package(HIP REQUIRED)
find_package(IntelSYCLForGCC REQUIRED)
find_package(XRT REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(cintroots REQUIRED)
find_package(compression REQUIRED)
find_package(libint2 REQUIRED)
pkg_search_module(hwloc REQUIRED IMPORTED_TARGET hwloc)

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

configure_file(
    include/constants.h.in
    include/constants.h
)
configure_file(
    include/fortran_interface.f08.in
    include/fortran_interface.f08
)

add_library(
    heric 
    SHARED 
        src/libheric.cpp include/libheric.h
        src/default_acc.cpp include/default_acc.h 
        src/cuda_acc.cpp include/cuda_acc.h
        src/hip_acc.cpp include/hip_acc.h
        src/oneapi_acc.cpp include/oneapi_acc.h
        src/xrt_acc.cpp include/xrt_acc.h
        src/quartet_input.cpp include/quartet_input.h
        src/transform.cpp include/transform.h
        src/util.cpp include/util.h
        include/fortran_interface.f08
)

target_include_directories(
    heric 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include> 
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/include
)
target_include_directories(
    heric
    SYSTEM PRIVATE
        ${XRT_INCLUDE_DIRS}
)
target_link_libraries(
    heric 
    PRIVATE 
        XRT::xrt_coreutil
        CUDA::cudart 
        hip::host
        IntelSYCLForGCC
        OpenMP::OpenMP_CXX
        dl
        Eigen3::Eigen
        cintroots::cintroots
        compression::compression
        Libint2::libint2
)

set_property(TARGET heric PROPERTY CXX_STANDARD 20)
target_compile_options(heric PUBLIC -mavx -mavx2)

add_executable(fortran_interface_test tests/fortran_interface_test.f08)
target_include_directories(
    fortran_interface_test 
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include"
)
target_link_libraries(fortran_interface_test PUBLIC heric MPI::MPI_Fortran OpenMP::OpenMP_Fortran)

#add_executable(hwloc_test tests/hwloc_test.cpp)
#target_link_libraries(hwloc_test PRIVATE PkgConfig::hwloc)
#
#add_executable(hwloc_mpi tests/hwloc_mpi.cpp)
#target_link_libraries(
#    hwloc_mpi 
#    PRIVATE
#        heric 
#        PkgConfig::hwloc
#        MPI::MPI_C
#)

install(
    TARGETS heric
    EXPORT  heric_targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES 
            ${CMAKE_CURRENT_BINARY_DIR}/libheric_interface.mod
        DESTINATION 
            ${CMAKE_INSTALL_INCLUDEDIR}) 

install(
    EXPORT heric_targets
    FILE heric_targets.cmake
    NAMESPACE heric::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/heric
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/heric-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/heric-config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/heric"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/heric-config-version.cmake"
    COMPATIBILITY AnyNewerVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/heric-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/heric-config-version.cmake"
    DESTINATION
        "${CMAKE_INSTALL_LIBDIR}/cmake/heric"
)

