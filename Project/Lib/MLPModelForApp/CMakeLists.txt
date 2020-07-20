cmake_minimum_required(VERSION 3.16)
project(MLPCppLib)

set(CMAKE_CXX_STANDARD 14)

add_library(MLPCppLib SHARED library.cpp library.h)