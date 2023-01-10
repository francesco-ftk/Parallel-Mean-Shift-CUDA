#include <cmath>

// Euclidean distance function
__device__ float l2Distance_cuda(float* row1, float* row2, size_t size) {
    float distance = 0;
    for (int i = 0; i < size; ++i) {
        distance += std::pow(row1[i] - row2[i], 2);
    }
    return sqrt(distance);
}
