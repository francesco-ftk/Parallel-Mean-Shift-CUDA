#ifndef MATRIX_MEANSHIFT_CPP
#define MATRIX_MEANSHIFT_CPP

#include <iostream>
#include "distance.cpp"

using namespace std;

/**
 * Cluster RGB RgbPixels with the mean shift algorithm
 *
 * The mean shift algorithm is used in a 5-dimensional space (r, g, b, x, y) to cluster the
 * RgbPixels of an image.
 *
 * @param points the structure of arrays containing the pixel values
 * @param modes the resulting modes to compute
 * @param bandwidth the radius of window size to compute the mean shift
 *
 * @todo
 * @return the array of cluster indices
 */
int matrixMeanShift(float* points, size_t nOfPoints, float bandwidth, size_t dimension, float* modes, int* clusters)
{
	// stop value to check for the shift convergence
	float epsilon = bandwidth * 0.05;

	// matrix to save the final mean of each pixel
	float means[nOfPoints * dimension];

	// compute the means
	for (int i = 0; i < nOfPoints; ++i) {
		//printf("  Examining point %d\n", i);

		// initialize the mean on the current point
		float mean[dimension];
		for (int k = 0; k < dimension; ++k) { mean[k] = points[i * dimension + k]; }

		// assignment to ensure the first computation
		float shift = epsilon;

		while (shift >= epsilon) {
			//printf("  iterating...\n");

			// initialize the centroid to 0, it will accumulate points later
			float centroid[dimension];
			for (int k = 0; k < dimension; ++k) { centroid[k] = 0; }

			// track the number of points inside the bandwidth window
			int windowPoints = 0;

			for (int j = 0; j < nOfPoints; ++j) {
				float point[dimension];
				for (int k = 0; k < dimension; ++k) { point[k] = points[j * dimension + k]; }

				if (l2Distance(mean, point, dimension) <= bandwidth) {
					// accumulate the point position
					for (int k = 0; k < dimension; ++k) {
						// todo: multiply by the chosen kernel
						centroid[k] += point[k];
					}
					++windowPoints;
				}
			}

			//printf("    %d points examined\n", windowPoints);

			// get the centroid dividing by the number of points taken into account
			for (int k = 0; k < dimension; ++k) { centroid[k] /= windowPoints; }

			shift = l2Distance(mean, centroid, dimension);

			//printf("    shift = %f\n", shift);

			// update the mean
			for (int k = 0; k < dimension; ++k) { mean[k] = centroid[k]; }
		}

		// mean now contains the mode of the point
		for (int k = 0; k < dimension; ++k) { means[i * dimension + k] = mean[k]; };
	}

	//printf("Meanshift: second phase start\n");

	// label all points as "not clustered"
	for (int k = 0; k < nOfPoints; ++k) { clusters[k] = -1; }

	// counter for the number of discovered clusters
	int clustersCount = 0;

	for (int i = 0; i < nOfPoints; ++i) {
		float mean[5];
		for (int k = 0; k < dimension; ++k) { mean[k] = means[i * dimension + k]; }

		/*printf("    Mean: [ ");
		for (int k = 0; k < dimension; ++k)
		{ printf("%f ", mean[k]); }
		printf("]\n");*/

		//printf("  Finding a cluster...\n");

		int j = 0;
		while (j < clustersCount && clusters[i] == -1)
		{
			// select the current mode
			float mode[dimension];
			for (int k = 0; k < dimension; ++k) { mode[k] = modes[j * dimension + k]; }

			// if the mean is close enough to the current mode
			if (l2Distance(mean, mode, dimension) < bandwidth)
			{
				//printf("    Cluster %d similar\n", j);

				/*printf("    Cluster: [ ");
				for (int k = 0; k < dimension; ++k)
				{ printf("%f ", mode[k]); }
				printf("]\n");*/

				// assign the point i to the cluster j
				clusters[i] = j;
			}
			++j;
		}
		// if the point i was not assigned to a cluster
		if (clusters[i] == -1) {
			//printf("    No similar clusters, creating a new one... (%d)", clustersCount);

			// create a new cluster associated with the mode of the point i
			clusters[i] = clustersCount;

			for (int k = 0; k < dimension; ++k) { modes[clustersCount * dimension + k] = mean[k]; }

			clustersCount++;
		}
	}

	//printf("Meanshift: end\n");
	return clustersCount;
}

#endif // MATRIX_MEANSHIFT_CPP