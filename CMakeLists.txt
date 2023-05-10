cmake_minimum_required(VERSION 3.19)
project(Parallel_Mean_Shift_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(cpp_sources)
add_subdirectory(cuda_sources)

add_executable(Parallel_Mean_Shift_CUDA main.cu)

set_target_properties(
        Parallel_Mean_Shift_CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(Parallel_Mean_Shift_CUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        #--generate-line-info
        --use_fast_math
        #--relocatable-device-code=true
        #-Xptxas="-v"
        >)