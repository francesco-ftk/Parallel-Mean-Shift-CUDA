#include <iostream>
#include <chrono>
#include "cuda_runtime.h"
#include "../cpp_sources/distance.cpp"
#include "errors.cu"
#include "distance.cu"

//#define BLOCKS 32
#define THREADS 512

using namespace std;
using namespace chrono;

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


// TODO portare in modalità ottimizzata a blocchi con shared
__global__ void matrixMeanShiftCUDA_kernel(float *points, size_t nOfPoints, float bandwidth, size_t dimension, float *means, int width, int height) {

    // stop value to check for the shift convergence
    float epsilon = bandwidth * 0.05;

    // compute the means
    unsigned int x = (blockIdx.x * blockDim.x + threadIdx.x) * dimension;
    if (x < nOfPoints * dimension) { // x <= nOfPoints*dimension-4
        float *mean = new float[dimension];
        // initialize the mean on the current point
        for (int k = 0; k < dimension; ++k) { mean[k] = points[x + k]; }

        // assignment to ensure the first computation
        float shift = epsilon;

        while (shift >= epsilon) {
            // initialize the centroid to 0, it will accumulate points later
            float *centroid = new float[dimension];
            for (int k = 0; k < dimension; ++k) { centroid[k] = 0; }

            // track the number of points inside the bandwidth window
            int windowPoints = 0;

            for (int j = 0; j < nOfPoints; ++j) {
                float *point = new float[dimension];
                for (int k = 0; k < dimension; ++k) { point[k] = points[j * dimension + k]; }

                if (l2Distance_cuda(mean, point, dimension) <= bandwidth) {
                    // accumulate the point position
                    for (int k = 0; k < dimension; ++k) {
                        // todo: multiply by the chosen kernel
                        centroid[k] += point[k];
                    }
                    ++windowPoints;
                }

                delete[] point;
            }

            // get the centroid dividing by the number of points taken into account
            for (int k = 0; k < dimension; ++k) { centroid[k] /= windowPoints; }

            shift = l2Distance_cuda(mean, centroid, dimension);

            // update the mean
            for (int k = 0; k < dimension; ++k) { mean[k] = centroid[k]; }

            delete[] centroid;
        }

        // mean now contains the mode of the point
        for (int k = 0; k < dimension; ++k) { means[x + k] = mean[k]; };

        delete[] mean;

    }
}

int matrixMeanShiftCUDA(float *points, size_t nOfPoints, float bandwidth, size_t dimension, float *modes, int *clusters, int width, int height) {

	int blocks = (int) ceil((float) nOfPoints / THREADS);

    float *dev_points = nullptr;
    // matrix to save the final mean of each pixel
    float *dev_means = nullptr;

    // Allocate GPU buffers for three arrays (two input, one output)
    CUDA_CHECK_RETURN(cudaMalloc((void **) &dev_points, nOfPoints * dimension * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &dev_means, nOfPoints * dimension * sizeof(float)));

    // Copy input arrays from host to device memory (global)
    CUDA_CHECK_RETURN(cudaMemcpy(dev_points, points, nOfPoints * dimension * sizeof(float), cudaMemcpyHostToDevice));

    auto start_time_cuda = high_resolution_clock::now();
    // Launch the kernel on the GPU
    matrixMeanShiftCUDA_kernel<<<blocks, THREADS>>>(dev_points, nOfPoints, bandwidth, dimension, dev_means, width, height);

    // Wait for the kernel to finish
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    auto end_time_cuda = high_resolution_clock::now();

    // Copy the result array from device to host memory
    float* means = new float[nOfPoints * dimension];
    CUDA_CHECK_RETURN(cudaMemcpy(means, dev_means, nOfPoints * dimension * sizeof(float), cudaMemcpyDeviceToHost));

    // Free the GPU buffers
    CUDA_CHECK_RETURN(cudaFree(dev_points));
    CUDA_CHECK_RETURN(cudaFree(dev_means));

    // TODO parte non imbarazzantemente parallela

	auto start_time_sequential = high_resolution_clock::now();

    // label all points as "not clustered"
    for (int k = 0; k < nOfPoints; ++k) { clusters[k] = -1; }

    // counter for the number of discovered clusters
    int clustersCount = 0;

    for (int i = 0; i < nOfPoints; ++i) {
		float* mean = new float[dimension];
        for (int k = 0; k < dimension; ++k) { mean[k] = means[i * dimension + k]; }

        int j = 0;
        while (j < clustersCount && clusters[i] == -1)
        {
            // select the current mode
			float* mode = new float[dimension];
            for (int k = 0; k < dimension; ++k) { mode[k] = modes[j * dimension + k]; }

            // if the mean is close enough to the current mode
            if (l2Distance(mean, mode, dimension) < bandwidth)
            {
                // assign the point i to the cluster j
                clusters[i] = j;
            }
            ++j;

			delete[] mode;
        }
        // if the point i was not assigned to a cluster
        if (clusters[i] == -1) {
            //printf("    No similar clusters, creating a new one... (%d)", clustersCount);

            // create a new cluster associated with the mode of the point i
            clusters[i] = clustersCount;

            for (int k = 0; k < dimension; ++k) { modes[clustersCount * dimension + k] = mean[k]; }

            clustersCount++;
        }

		delete[] mean;
    }

	auto end_time_sequential = high_resolution_clock::now();

	// timings
	float totalTime_cuda = duration_cast<microseconds>(end_time_cuda - start_time_cuda).count() / 1000.f;
	float totalTime_sequential = duration_cast<microseconds>(end_time_sequential - start_time_sequential).count() / 1000.f;
	float totalTime = totalTime_cuda + totalTime_sequential;

	printf("Cuda timings:");
	printf("  cuda:   %fms\n", totalTime_cuda);
	printf("  sequential: %fms\n", totalTime_sequential);
	printf("  total: %fms\n", totalTime);

    return clustersCount;

}

