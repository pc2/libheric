include(CMakePrintHelpers)


find_library(_SYCL_SO_FILE NAMES libsycl.so REQUIRED PATH_SUFFIXES "../lib")

cmake_path(GET _SYCL_SO_FILE PARENT_PATH _SYCL_LIB_PATH )
cmake_path(GET _SYCL_LIB_PATH PARENT_PATH _SYCL_ROOT_PATH )
cmake_path(APPEND _SYCL_ROOT_PATH include OUTPUT_VARIABLE _SYCL_INCLUDE_PATH)

#message("_SYCL_SO_FILE = ${_SYCL_SO_FILE}")
#message("_SYCL_LIB_PATH = ${_SYCL_LIB_PATH}")
#message("_SYCL_ROOT_PATH = ${_SYCL_ROOT_PATH}")

if(NOT _SYCL_ROOT_PATH MATCHES "oneapi")
    message(FATAL_ERROR "SYCL implementation (${_SYCL_ROOT_PATH}) is not oneapi specific")
endif()

# Make sure FPGA extensions are available
find_file(_SYCL_FPGA_EXT_FILE NAMES fpga_extensions.hpp PATHS ${_SYCL_ROOT_PATH} PATH_SUFFIXES "include/sycl/ext/intel" REQUIRED)  

add_library(IntelSYCLForGCC SHARED IMPORTED)
target_include_directories(
    IntelSYCLForGCC 
    INTERFACE
        "${_SYCL_INCLUDE_PATH}" 
        "${_SYCL_INCLUDE_PATH}/sycl"
)

target_link_directories(
    IntelSYCLForGCC
    INTERFACE
        "${_SYCL_LIB_PATH}"
)

target_link_libraries(
    IntelSYCLForGCC
    INTERFACE
        sycl
)

set_target_properties(
    IntelSYCLForGCC 
    PROPERTIES 
    IMPORTED_LOCATION "${_SYCL_SO_FILE}"
)

cmake_print_properties(
    TARGETS IntelSYCLForGCC
    PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES
        INTERFACE_LINK_DIRECTORIES
		INTERFACE_LINK_LIBRARIES
        IMPORTED_LOCATION
)
