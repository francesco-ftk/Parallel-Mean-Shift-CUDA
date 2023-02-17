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

	float h1 = row1[0];
	float h2 = row2[0];
	float s1 = row1[1];
	float s2 = row2[1];
	float x1 = s1 * std::cos(h1);
	float x2 = s2 * std::cos(h2);
	float y1 = s1 * std::sin(h1);
	float y2 = s2 * std::sin(h2);
	squaredDistance += pow(x1 - x2, 2);
	squaredDistance += pow(y1 - y2, 2);

	//float absDifference = abs(row1[0]-row2[0]);
    //squaredDistance += std::pow(min(absDifference, 1 - absDifference), 2);
    for (int i = 2; i < size; ++i) {
        squaredDistance += std::pow(row1[i] - row2[i], 2);
    }
    return squaredDistance;
}
