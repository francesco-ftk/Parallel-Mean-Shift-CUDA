#ifndef DISTANCE_CPP
#define DISTANCE_CPP

#include <cmath>

// Euclidean distance function
float l2Distance(float* row1, float* row2, size_t size) {
	float distance = 0;
	for (int i = 0; i < size; ++i) {
		distance += std::pow(row1[i] - row2[i], 2);
	}
	return sqrt(distance);
}

#endif // DISTANCE_CPP