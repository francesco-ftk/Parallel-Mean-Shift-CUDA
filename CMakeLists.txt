cmake_minimum_required(VERSION 3.23)
project(Parallel_Mean_Shift_CUDA CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Parallel_Mean_Shift_CUDA main.cu)

set_target_properties(Parallel_Mean_Shift_CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
