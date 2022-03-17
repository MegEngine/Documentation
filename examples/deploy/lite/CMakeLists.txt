project(LiteExmaple)
cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PROJECT_BINARY_DIR ${PROJECT_SOURCE_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT DEFINED ENV{LITE_INSTALL_DIR} AND NOT LITE_INSTALL_DIR)
  message(FATAL_ERROR "LITE_INSTALL_DIR is not defined.")
endif()

find_package(OpenCV REQUIRED)

include_directories("$ENV{LITE_INSTALL_DIR}/include" ${OpenCV_INCLUDE_DIRS})
link_directories("$ENV{LITE_INSTALL_DIR}/lib/x86_64" ${OpencV_LIBRARY_DIRS})

# Softmax test
add_executable(lite_softmax main.cpp)
target_link_libraries(lite_softmax lite_shared)

# Image inference
add_executable(inference inference.cpp)
target_link_libraries(inference ${OpenCV_LIBS} lite_shared)
