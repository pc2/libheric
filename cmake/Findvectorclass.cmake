include(CMakePrintHelpers)
find_file(_VECTORCLASS_H_FILE NAMES vectorclass.h REQUIRED)

cmake_path(GET _VECTORCLASS_H_FILE PARENT_PATH _VECTORCLASS_PATH )

#message("_SYCL_SO_FILE = ${_SYCL_SO_FILE}")
#message("_SYCL_LIB_PATH = ${_SYCL_LIB_PATH}")
#message("_SYCL_ROOT_PATH = ${_SYCL_ROOT_PATH}")

if(NOT _VECTORCLASS_PATH MATCHES "version2")
    message(FATAL_ERROR "vectorclass.h is not from version2")
endif()

add_library(vectorclass INTERFACE IMPORTED)
target_include_directories(
    vectorclass 
    INTERFACE
        "${_VECTORCLASS_PATH}" 
)

cmake_print_properties(
    TARGETS vectorclass
    PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES
        INTERFACE_LINK_DIRECTORIES
		INTERFACE_LINK_LIBRARIES
        IMPORTED_LOCATION
)
