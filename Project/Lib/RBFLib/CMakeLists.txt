cmake_minimum_required(VERSION 3.16)
project(RBFLib)

set(CMAKE_CXX_STANDARD 14)
set(lib_EIGEN_DIR "C:/Users/hejar/Eigen")

include_directories(${lib_EIGEN_DIR})


add_library(RBFLib SHARED library.cpp library.h)

target_link_libraries(RBFLib "C:/Users/hejar/Eigen")