#include <cmath>

// Euclidean distance function
__device__ float l2Distance_cuda(float* row1, float* row2, size_t size) {
    float distance = 0;
    for (int i = 0; i < size; ++i) {
        distance += std::pow(row1[i] - row2[i], 2);
    }
    return sqrt(distance);
}

__device__ float l2SquaredDistance_cuda(float* row1, float* row2, size_t size) {
	float squaredDistance = 0;
	for (int i = 0; i < size; ++i) {
		squaredDistance += std::pow(row1[i] - row2[i], 2);
	}
	return squaredDistance;
}

// HSV Euclidean distance
__device__ float l2_HSV_Distance_cuda(float* row1, float* row2, size_t size){
    float squaredDistance = 0;
    squaredDistance += std::pow(min(abs(row1[0]-row2[0]), 1-abs(row1[0]-row2[0])),2);
    for (int i = 1; i < size; ++i) {
        squaredDistance += std::pow(row1[i] - row2[i], 2);
    }
    return squaredDistance;
}
