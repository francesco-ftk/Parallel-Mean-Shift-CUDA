#ifndef SOA_MEANSHIFT_CPP
#define SOA_MEANSHIFT_CPP

#include "rgb_pixels.cpp"
#include "distance.cpp"
#include <iostream>

/**
 * Cluster RGB pixels with the mean shift algorithm
 *
 * The mean shift algorithm is used in a 5-dimensional space (R, G, B, X, Y) to cluster the pixels of an image.
 *
 * @param pixels the structure of arrays containing the pixel values
 * @param n the number of pixels
 * @param bandwidth the radius of the window size used to select the points to take into account
 * @param modes (output) the SoA containing the computed modes (size must be n)
 * @param clusters (output) an index array that associates each pixel to its mode (size must be n)
 *
 * @return the array of cluster indices
 */
int soaMeanShift(RgbPixels &points, size_t nOfPoints, float bandwidth, RgbPixels &modes, int* clusters)
{
	int dimension = RgbPixels::SPACE_DIMENSION;

	// sanity check
	if (&points == &modes) {
		printf("Error - Pixel and modes can't be the same structure!");
		return -1;
	}

	// stop value to check for the shift convergence
	float epsilon = bandwidth * 0.05;

	// structure of array to save the final mean of each pixel
	RgbPixels means;
	means.create(points.width, points.height);

	//printf("Meanshift: first phase start\nOfPoints");

	// compute the means
	for (int i = 0; i < nOfPoints; ++i) {
		//printf("  Examining point %d\n", i);

		// initialize the mean on the current point
		float* mean = new float[dimension];
		points.write(i, mean);

		// assignment to ensure the first computation
		float shift = epsilon;

		while (shift >= epsilon) {
			//printf("  iterating...\n");

			// initialize the centroid to 0, it will accumulate points later
			float* centroid = new float[dimension];
			for (int k = 0; k < 5; ++k) { centroid[k] = 0; }

			// track the number of points inside the bandwidth window
			int windowPoints = 0;

			for (int j = 0; j < nOfPoints; ++j) {
				float* point = new float[dimension];
				points.write(j, point);

				if (l2Distance(mean, point, dimension) <= bandwidth) {
					// accumulate the point position
					for (int k = 0; k < dimension; ++k) {
						// todo: multiply by the chosen kernel
						centroid[k] += point[k];
					}
					++windowPoints;
				}

				delete[] point;
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
		means.save(mean, i);
	}

	//printf("Meanshift: second phase start\n");

	// label all points as "not clustered"
	for (int k = 0; k < nOfPoints; ++k) { clusters[k] = -1; }

	// counter for the number of discovered clusters
	int clustersCount = 0;

	for (int i = 0; i < nOfPoints; ++i) {
		float mean[5];
		means.write(i, mean);

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
			modes.write(j, mode);

			// if the mean is close enough to the current mode
			if(l2Distance(mean, mode, dimension) < bandwidth)
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

			modes.save(mean, clustersCount);

			clustersCount++;
		}
	}

	//printf("Meanshift: end\n");
	return clustersCount;
}

#endif // SOA_MEANSHIFT_CPP