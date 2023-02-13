#ifndef DISTANCE_CPP
#define DISTANCE_CPP

#include <cmath>
#include <algorithm>

// Euclidean distance function
float l2Distance(float* row1, float* row2, size_t size) {
	float distance = 0;
	for (int i = 0; i < size; ++i) {
		distance += std::pow(row1[i] - row2[i], 2);
	}
	return sqrt(distance);
}

// HSV Euclidean distance function
float l2_HSV_Distance(float* row1, float* row2, size_t size){
    float squaredDistance = 0;
    squaredDistance += std::pow(std::min(abs(row1[0]-row2[0]), 1-abs(row1[0]-row2[0])),2);
    for (int i = 1; i < size; ++i) {
        squaredDistance += std::pow(row1[i] - row2[i], 2);
    }
    return squaredDistance;
}


#endif // DISTANCE_CPP