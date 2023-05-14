cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 14)

add_library(source_cpp
        matrix_meanshift.cpp
        distance.cpp
        ppm_io.h
        ppm_io.cpp
        )