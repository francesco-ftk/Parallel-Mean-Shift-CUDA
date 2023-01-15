cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 14)

add_library(source_cpp
        distance.cpp
        matrix_meanshift.cpp
        ppm_io.h
        ppm_io.cpp
        rgb_pixels.cpp
        soa_meanshift.cpp)