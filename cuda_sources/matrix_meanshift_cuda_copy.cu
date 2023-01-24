#include <iostream>
#include <chrono>
#include "../cpp_sources/distance.cpp"
#include "errors.cu"
#include "distance.cu"

#define CHANNELS 5
#define EPSILON_MULTIPLIER 0.05f
#define THREADS_X 16
#define THREADS_Y 16
#define TILE_WIDTH 16

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

__constant__ float const_bandwidth;

__global__ void matrixMeanShiftCUDA_kernel(const float *points, float *means, int width, int height)
{
	__shared__ float pointsTile[TILE_WIDTH][TILE_WIDTH * CHANNELS];
	__shared__ bool tileShift;

	unsigned bx = blockIdx.x; unsigned tx = threadIdx.x;
	unsigned by = blockIdx.y; unsigned ty = threadIdx.y;

	unsigned int row = by * blockDim.y + ty;
	unsigned int col = bx * blockDim.x + tx;
	unsigned int pos = (row * width + col) * CHANNELS;

	unsigned int phasesX = ceil((float) width / TILE_WIDTH);
	unsigned int phasesY = ceil((float) height / TILE_WIDTH);

    // stop value to check for the shift convergence
    float epsilon = const_bandwidth * EPSILON_MULTIPLIER;

    // check if the thread pixel is not outside the image
	if (row < height && col < width)
	{
        float* mean = &means[pos];
		//float* mean = new float[CHANNELS]; // TODO test if faster

        // initialize the mean on the current point
        for (int k = 0; k < CHANNELS; ++k) { mean[k] = points[pos + k]; }

        // assignment to ensure the first computation
        float shift = epsilon;

		auto* centroid = new float[CHANNELS];

		// tileShift is true if at least one thread per block should continue
        while (shift >= epsilon || tileShift)
		{
			// reset tileShift
			//if (tx == 0 && ty == 0) {
				tileShift = false;
			//}

			// track the number of points inside the const_bandwidth window
			int windowPoints = 0;

			// initialize the centroid to 0 to accumulate points later
			for (int k = 0; k < CHANNELS; ++k) { centroid[k] = 0; }

			for (int phaseY = 0; phaseY < phasesY; ++phaseY)
			{
				for (int phaseX = 0; phaseX < phasesX; ++phaseX)
				{
					// fixme tile size must be <= block size
					// todo optimize (1 thread per channel)
					if (ty < TILE_WIDTH && tx < TILE_WIDTH)
					{
						unsigned int phaseRow = phaseY * TILE_WIDTH + row % TILE_WIDTH;
						unsigned int phaseCol = phaseX * TILE_WIDTH + col % TILE_WIDTH;
						unsigned int phasePos = (phaseRow * width + phaseCol) * CHANNELS;

						for (int k = 0; k < CHANNELS; ++k) { pointsTile[ty][tx * CHANNELS + k] = points[phasePos + k]; }
					}
					__syncthreads();

					int tileDimX = min(TILE_WIDTH, width - TILE_WIDTH * phaseX);
					int tileDimY = min(TILE_WIDTH, height - TILE_WIDTH * phaseY);

					for (int tileRow = 0; tileRow < tileDimY; ++tileRow) {
						for (int tileCol = 0; tileCol < tileDimX; ++tileCol) {
							//float point[CHANNELS];
							float *point = &pointsTile[tileRow][tileCol * CHANNELS];
							//for (int k = 0; k < CHANNELS; ++k) { point[k] = pointsTile[tileRow * CHANNELS + k]; }

							if (l2Distance_cuda(mean, point, CHANNELS) <= const_bandwidth) {
								// accumulate the point position
								for (int k = 0; k < CHANNELS; ++k) {
									// todo: multiply by the chosen kernel
									centroid[k] += point[k];
								}
								++windowPoints;
							}
						}
					}
					__syncthreads();
				}
			}

			// get the centroid dividing by the number of points taken into account
			for (int k = 0; k < CHANNELS; ++k) { centroid[k] /= windowPoints; }

			shift = l2Distance_cuda(mean, centroid, CHANNELS);
			//shift = 0; // FIXME

			// update the mean
			for (int k = 0; k < CHANNELS; ++k) { mean[k] = centroid[k]; }

			if (shift >= epsilon) {
				tileShift = true;
			}

			__syncthreads();
        }

		/*if (tx == 0 && ty == 0) {
			printf("Block (%d,%d) iterations: %d\n", bx, by, tmp);
		}*/

		// TODO test me
		//for (int k = 0; k < CHANNELS; ++k) { means[x + k] = mean[k]; }
		//delete[] mean;

		delete[] centroid;

        /*
        clock_block(1000);
        printf("first_clock_finished %d block %d\n", threadIdx.x, blockIdx.x);
        clock_block(1000);
        printf("second_clock_finished %d block %d \n", threadIdx.x, blockIdx.x);
        */

        // mean now contains the mode of the point
        //for (int k = 0; k < CHANNELS; ++k) { means[x + k] = mean[k]; };
    }
}

int matrixMeanShiftCUDA(float *points, float bandwidth, size_t dimension, float *modes, int *clusters, int width, int height) {

	int nOfPoints = width * height;
	int gridSizeX = (int) ceil((float) width / THREADS_X);  // 7
	int gridSizeY = (int) ceil((float) height / THREADS_Y); // 7
	dim3 gridSize(gridSizeX, gridSizeY, 1);
	dim3 blockSize(THREADS_X, THREADS_Y, 1);

	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(const_bandwidth, &bandwidth, sizeof(float)));

    float *dev_points = nullptr;
    // matrix to save the final mean of each pixel
    float *dev_means = nullptr;

    // Allocate GPU buffers for three arrays (two input, one output)
    CUDA_CHECK_RETURN(cudaMalloc((void **) &dev_points, nOfPoints * dimension * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &dev_means, nOfPoints * dimension * sizeof(float)));

    // Copy input arrays from host to device memory (global)
    CUDA_CHECK_RETURN(cudaMemcpy(dev_points, points, nOfPoints * dimension * sizeof(float), cudaMemcpyHostToDevice));

    printf("call kernel...\n");
    auto start_time_cuda = high_resolution_clock::now();
    // Launch the kernel on the GPU
    matrixMeanShiftCUDA_kernel<<<gridSize, blockSize>>>(dev_points, dev_means, width, height);
    //matrixMeanShiftCUDA_kernel<<<10, 64>>>(dev_points, nOfPoints, const_bandwidth, dimension, dev_means, width, height);

    // Wait for the kernel to finish
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    auto end_time_cuda = high_resolution_clock::now();

    printf("out from kernel \n");

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


