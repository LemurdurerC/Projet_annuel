cmake_minimum_required(VERSION 3.16)
project(SVM1)

set(CMAKE_CXX_STANDARD 14)

set(lib_EIGEN_DIR "C:/Users/hejar/Eigen")

include_directories(${lib_EIGEN_DIR})

add_executable(SVM1 main.cpp)

target_link_libraries(SVM1 "C:/Users/hejar/Eigen")