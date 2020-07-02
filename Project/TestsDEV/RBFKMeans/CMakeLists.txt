cmake_minimum_required(VERSION 3.16)
project(RBFKMeans)

set(CMAKE_CXX_STANDARD 14)

set(lib_EIGEN_DIR "C:/Users/hejar/Eigen")

include_directories(${lib_EIGEN_DIR})


add_executable(RBFKMeans main.cpp)

target_link_libraries(RBFKMeans "C:/Users/hejar/Eigen")