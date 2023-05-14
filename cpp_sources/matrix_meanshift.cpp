#ifndef MATRIX_MEANSHIFT_CPP
#define MATRIX_MEANSHIFT_CPP

#define CHANNELS 5

#include "distance.cpp"

int matrixMeanShift(float* points, size_t nOfPoints, float bandwidth, float* modes, int* clusters)
{
	auto squaredBandwidth = (float) pow(bandwidth, 2);

	// stop value to check for the shift convergence
	auto epsilon = (float) pow(bandwidth * 0.05, 2);

	// matrix to save the final mean of each pixel
	auto means = new float[nOfPoints * CHANNELS];

	// compute the means
	for (int i = 0; i < nOfPoints; ++i)
	{
		// initialize the mean on the current point
		float mean[CHANNELS];
		for (int k = 0; k < CHANNELS; ++k) { mean[k] = points[i * CHANNELS + k]; }

		// assignment to ensure the first computation
		float shift = epsilon;

		while (shift >= epsilon)
		{
			// initialize the centroid to 0, it will accumulate points later
			float centroid[CHANNELS];
			for (float& k : centroid) { k = 0; }

			// track the number of points inside the bandwidth window
			int windowPoints = 0;

			for (int j = 0; j < nOfPoints; ++j)
			{
				float point[CHANNELS];
				for (int k = 0; k < CHANNELS; ++k) { point[k] = points[j * CHANNELS + k]; }

				if (l2SquaredDistance(mean, point, CHANNELS) <= squaredBandwidth)
				{
					// accumulate the point position
					for (int k = 0; k < CHANNELS; ++k)
					{
						centroid[k] += point[k];
					}
					++windowPoints;
				}
			}

			// get the centroid dividing by the number of points taken into account
			for (float& k : centroid) { k /= (float) windowPoints; }

			shift = l2SquaredDistance(mean, centroid, CHANNELS);

			// update the mean
			for (int k = 0; k < CHANNELS; ++k) { mean[k] = centroid[k]; }
		}

		// mean now contains the mode of the point
		for (int k = 0; k < CHANNELS; ++k) { means[i * CHANNELS + k] = mean[k]; };
	}

	// label all points as "not clustered"
	for (int k = 0; k < nOfPoints; ++k) { clusters[k] = -1; }

	// counter for the number of discovered clusters
	int clustersCount = 0;

	for (int i = 0; i < nOfPoints; ++i)
	{
		float mean[CHANNELS];
		for (int k = 0; k < CHANNELS; ++k) { mean[k] = means[i * CHANNELS + k]; }

		int j = 0;
		while (j < clustersCount && clusters[i] == -1)
		{
			// select the current mode
			float mode[CHANNELS];
			for (int k = 0; k < CHANNELS; ++k) { mode[k] = modes[j * CHANNELS + k]; }

			// if the mean is close enough to the current mode
			if (l2SquaredDistance(mean, mode, CHANNELS) < squaredBandwidth)
			{
				// assign the point i to the cluster j
				clusters[i] = j;
			}
			++j;
		}
		// if the point i was not assigned to a cluster
		if (clusters[i] == -1) {
			// create a new cluster associated with the mode of the point i
			clusters[i] = clustersCount;

			for (int k = 0; k < CHANNELS; ++k) { modes[clustersCount * CHANNELS + k] = mean[k]; }

			clustersCount++;
		}
	}

	delete[] means;

	return clustersCount;
}

#endif // MATRIX_MEANSHIFT_CPP