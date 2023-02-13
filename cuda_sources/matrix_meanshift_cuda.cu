#include <iostream>
#include <chrono>
#include "../cpp_sources/distance.cpp"
#include "errors.cu"
#include "distance.cu"

#define CHANNELS 5
#define EPSILON_MULTIPLIER 0.125f // this is different because bandwidth is squared
#define THREADS_X 16
#define THREADS_Y 16
#define TILE_WIDTH 16

using namespace std;
using namespace chrono;

// TODO kernel to weight the sums
// TODO check with Excel sheet (shared memory)
// TODO optimize to get coalesced access to memory
// TODO update after HSV change
/**
 * CUDA kernel to cluster an RGB image with the mean shift algorithm
 *
 * The mean shift algorithm is used in a 5-dimensional space (r, g, b, x, y) to cluster the pixels of an image.
 *
 * @param points the pixel values of the image
 * @param means the output array of means to compute
 * @param width the width of the image
 * @param height the height of the image
 *
 * @param bandwidth
 */

__constant__ float const_squaredBandwidth;

__global__ void matrixMeanShiftCUDA_kernel(const float *points, float *means, int width, int height)
{
	__shared__ float shared_tile[TILE_WIDTH][TILE_WIDTH * CHANNELS];
	__shared__ bool shared_continueIteration;

	unsigned bx = blockIdx.x; unsigned tx = threadIdx.x;
	unsigned by = blockIdx.y; unsigned ty = threadIdx.y;

	unsigned int row = by * blockDim.y + ty;
	unsigned int col = bx * blockDim.x + tx;
	unsigned int pos = (row * width + col) * CHANNELS;

	unsigned int phasesX = ceil((float) width / TILE_WIDTH);
	unsigned int phasesY = ceil((float) height / TILE_WIDTH);

    // stop value to check for the shift convergence
    float epsilon = const_squaredBandwidth * EPSILON_MULTIPLIER;

	float* mean;
	auto* centroid = new float[CHANNELS];

    // check if the thread pixel is not outside the image
	if (row < height && col < width) {
		mean = &means[pos];

		// initialize the mean on the current point
		for (int k = 0; k < CHANNELS; ++k) { mean[k] = points[pos + k]; }
	}

	// set to ensure the first computation
	atomicOr((int*) &shared_continueIteration, true);

	// shared_continueIteration is true if at least one thread per block must continue
	while (shared_continueIteration)
	{
		// track the number of points inside the const_squaredBandwidth window
		int windowPoints = 0;

		// initialize the centroid to 0 to accumulate points later
		for (int k = 0; k < CHANNELS; ++k) { centroid[k] = 0; }

		for (int phaseY = 0; phaseY < phasesY; ++phaseY)
		{
			for (int phaseX = 0; phaseX < phasesX; ++phaseX)
			{
				int tileDimX = min(TILE_WIDTH, width - TILE_WIDTH * phaseX);
				int tileDimY = min(TILE_WIDTH, height - TILE_WIDTH * phaseY);

				// FIXME use 2-batch loading (14_gpu_cuda_6 slide 6)
				// TODO optimize (1 thread per channel)

				// load shared memory
				if (ty < tileDimY && tx < tileDimX)
				{
					unsigned int phaseRow = phaseY * TILE_WIDTH + row % TILE_WIDTH;
					unsigned int phaseCol = phaseX * TILE_WIDTH + col % TILE_WIDTH;
					unsigned int phasePos = (phaseRow * width + phaseCol) * CHANNELS;

					for (int k = 0; k < CHANNELS; ++k) { shared_tile[ty][tx * CHANNELS + k] = points[phasePos + k]; }
				}

				__syncthreads();

				// compute the mean
				if (row < height && col < width)
				{
					for (int tileRow = 0; tileRow < tileDimY; ++tileRow)
					{
						for (int tileCol = 0; tileCol < tileDimX; ++tileCol)
						{
							float *point = &shared_tile[tileRow][tileCol * CHANNELS];

							if (l2_HSV_Distance_cuda(mean, point, CHANNELS) <= const_squaredBandwidth)
							{
								// accumulate the point position
								for (int k = 0; k < CHANNELS; ++k)
								{
									centroid[k] += point[k];
								}
								++windowPoints;
							}
						}
					}
				}

				__syncthreads();
			}
		}

		// reset
		atomicAnd((int*) &shared_continueIteration, false);

		__syncthreads();

		// check if the thread pixel is not outside the image
		if (row < height && col < width) {
			// get the centroid dividing by the number of points taken into account
			for (int k = 0; k < CHANNELS; ++k) { centroid[k] /= (float) windowPoints; }

			float shift = l2_HSV_Distance_cuda(mean, centroid, CHANNELS);

			// update the mean
			for (int k = 0; k < CHANNELS; ++k) { mean[k] = centroid[k]; }

			// set if the thread must continue, hence the block
			if (shift >= epsilon) {
				atomicOr((int *) &shared_continueIteration, true);
			}
		}

		__syncthreads();
	}

	delete[] centroid;
}

int matrixMeanShiftCUDA(float *points, float bandwidth, size_t dimension, float *modes, int *clusters, int width, int height) {

	int nOfPoints = width * height;
	int gridSizeX = (int) ceil((float) width / THREADS_X);  // 7
	int gridSizeY = (int) ceil((float) height / THREADS_Y); // 7
	dim3 gridSize(gridSizeX, gridSizeY, 1);
	dim3 blockSize(THREADS_X, THREADS_Y, 1);

	auto squaredBandwidth = (float) std::pow(bandwidth, 2);

	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(const_squaredBandwidth, &squaredBandwidth, sizeof(float)));

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
    matrixMeanShiftCUDA_kernel<<<gridSize, blockSize>>>(dev_points, dev_means, width, height);

    // Wait for the kernel to finish
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    auto end_time_cuda = high_resolution_clock::now();

    // Copy the result array from device to host memory
    auto* means = new float[nOfPoints * dimension];
    CUDA_CHECK_RETURN(cudaMemcpy(means, dev_means, nOfPoints * dimension * sizeof(float), cudaMemcpyDeviceToHost));

    // Free the GPU buffers
    CUDA_CHECK_RETURN(cudaFree(dev_points));
    CUDA_CHECK_RETURN(cudaFree(dev_means));

    // sequential phase

    auto start_time_sequential = high_resolution_clock::now();

    // label all points as "not clustered"
    for (int k = 0; k < nOfPoints; ++k) { clusters[k] = -1; }

    // counter for the number of discovered clusters
    int clustersCount = 0;

    for (int i = 0; i < nOfPoints; ++i) {
        auto* mean = new float[dimension];
        for (int k = 0; k < dimension; ++k) { mean[k] = means[i * dimension + k]; }

        int j = 0;
        while (j < clustersCount && clusters[i] == -1)
        {
            // select the current mode
            auto* mode = new float[dimension];
            for (int k = 0; k < dimension; ++k) { mode[k] = modes[j * dimension + k]; }

            // if the mean is close enough to the current mode
            if (l2_HSV_Distance(mean, mode, dimension) < squaredBandwidth)
            {
                // assign the point i to the cluster j
                clusters[i] = j;
            }
            ++j;

            delete[] mode;
        }

        // if the point i was not assigned to a cluster
        if (clusters[i] == -1) {

            // create a new cluster associated with the mode of the point i
            clusters[i] = clustersCount;

            for (int k = 0; k < dimension; ++k) { modes[clustersCount * dimension + k] = mean[k]; }

            clustersCount++;
        }

        delete[] mean;
    }

    auto end_time_sequential = high_resolution_clock::now();

    // timings
    float totalTime_cuda = (float) duration_cast<microseconds>(end_time_cuda - start_time_cuda).count() / 1000.f;
    float totalTime_sequential = (float) duration_cast<microseconds>(end_time_sequential - start_time_sequential).count() / 1000.f;
    float totalTime = totalTime_cuda + totalTime_sequential;

    printf("Cuda timings:");
    printf("  cuda:   %fms\n", totalTime_cuda);
    printf("  sequential: %fms\n", totalTime_sequential);
    printf("  total: %fms\n", totalTime);

    return clustersCount;
}


