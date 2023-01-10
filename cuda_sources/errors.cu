#pragma once

#include <iostream>
#include <string>

#define CUDA_CHECK_RETURN(value) checkCudaError(__FILE__,__LINE__, #value, value)

void checkCudaError(const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess) {
		return;
	}

	std::cerr << statement << " returned " << cudaGetErrorString(err) <<
			  "(" << cudaGetErrorName(err) << ") at " << file << ":" << std::to_string(line) << std::endl;
	exit (1);
}