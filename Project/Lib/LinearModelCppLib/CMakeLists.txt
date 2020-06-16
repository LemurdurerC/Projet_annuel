cmake_minimum_required(VERSION 3.16)
project(LinearModelCppLib)

set(CMAKE_CXX_STANDARD 14)

set(lib_EIGEN_DIR "C:/Users/hejar/Eigen")

include_directories(${lib_EIGEN_DIR})


add_library(LinearModelCppLib SHARED library.cpp library.h)

target_link_libraries(LinearModelCppLib "C:/Users/hejar/Eigen")