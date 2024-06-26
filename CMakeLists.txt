cmake_minimum_required(VERSION 3.10)

project(MKLTests)
find_package(MKL CONFIG REQUIRED PATHS /home/lucas.reis/intel/oneapi/mkl/latest)

add_compile_definitions(MKL_DIRECT_CALL)

add_library(common STATIC common.c)
target_include_directories(common PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_options(common PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_link_libraries(common PUBLIC $<LINK_ONLY:MKL::MKL>)

file(GLOB BENCH_LIST "*.cpp")
foreach(BENCH_FILE ${BENCH_LIST})
  get_filename_component(BENCH_NAME ${BENCH_FILE} NAME_WLE)
  add_executable(${BENCH_NAME} ${BENCH_FILE})
  add_executable(${BENCH_NAME}_store_output ${BENCH_FILE})
  target_link_libraries(${BENCH_NAME} PUBLIC common)
  target_link_libraries(${BENCH_NAME}_store_output PUBLIC common)

  target_compile_definitions(${BENCH_NAME}_store_output PUBLIC MEASURE_ERROR RANDOM_INPUTS)
endforeach()
