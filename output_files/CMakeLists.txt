cmake_minimum_required(VERSION 3.10)

project(COMPARE_OUTPUTS)
find_package(MKL CONFIG REQUIRED PATHS /home/lucas.reis/intel/oneapi/mkl/latest)

add_compile_definitions(MKL_DIRECT_CALL)

set(CMAKE_CXX_STANDARD 20)
add_library(common STATIC IMPORTED)
set_target_properties(common PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/../../build/libcommon.a)

add_executable(compare_outputs compare_outputs.cpp)

target_include_directories(compare_outputs PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_options(compare_outputs PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_link_libraries(compare_outputs PUBLIC common $<LINK_ONLY:MKL::MKL>)
