cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 14)

add_library(source_cu
        matrix_meanshift_cuda.cu
        matrix_meanshift_cuda_copy.cu
        soa_meanshift_cuda.cu
        errors.cu
        distance.cu)