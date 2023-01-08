#pragma once

#include <iostream>

float cuSum(float a, float b) {
	return a + b;
}

__global__ void greetFromGpu() {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Hello from GPU thread %d\n", idx);
}