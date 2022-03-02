project(LiteExmaple)
cmake_minimum_required(VERSION 3.15)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LiteExmaple_SOURCE_DIR}/bin)

if(NOT DEFINED ENV{LITE_INSTALL_DIR})
  message(FATAL_ERROR "Env LITE_INSTALL_DIR not defined.")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV REQUIRED)
include_directories()

include_directories("$ENV{LITE_INSTALL_DIR}/include" ${OpenCV_INCLUDE_DIRS})
link_directories("$ENV{LITE_INSTALL_DIR}/lib/x86_64" ${OpencV_LIBRARY_DIRS})

# Softmax test
add_executable(lite_softmax main.cpp)
target_link_libraries(lite_softmax lite_shared)

# Image inference
add_executable(inference inference.cpp)
target_link_libraries(inference ${OpenCV_LIBS} lite_shared)
