cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 14)

add_library(source_cu
        matrix_meanshift_cuda.cu
        errors.cu
        distance.cu
        color_converter.cu)