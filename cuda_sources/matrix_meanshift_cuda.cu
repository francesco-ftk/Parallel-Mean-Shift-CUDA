#include <iostream>
#include <chrono>
#include "../cpp_sources/distance.cpp"
#include "errors.cu"
#include "distance.cu"

#define CHANNELS 5
#define EPSILON_MULTIPLIER 0.05f // this is different because bandwidth is squared
#define THREADS_X 16
#define THREADS_Y 16
#define TILE_WIDTH 16

// TODO: kernel to weight the sums

__constant__ float const_squaredBandwidth;

__global__ void matrixMeanShiftCUDA_kernel(const float *points, float *means, int width, int height)
{
	__shared__ float shared_tile[TILE_WIDTH][TILE_WIDTH * CHANNELS];
	__shared__ bool shared_continueIteration;
	bool private_continueIteration = true;

	unsigned bx = blockIdx.x; unsigned tx = threadIdx.x;
	unsigned by = blockIdx.y; unsigned ty = threadIdx.y;

	unsigned int row = by * blockDim.y + ty;
	unsigned int col = bx * blockDim.x + tx;
	unsigned int pos = (row * width + col) * CHANNELS;

	unsigned int phasesX = ceil((float) width / TILE_WIDTH);
	unsigned int phasesY = ceil((float) height / TILE_WIDTH);

    // stop value to check for the shift convergence
    auto epsilon = (float) pow(sqrt(const_squaredBandwidth) * EPSILON_MULTIPLIER, 2);

	float mean[CHANNELS];

    // check if the thread pixel is not outside the image
	if (row < height && col < width) {
		// initialize the mean on the current point
		for (int k = 0; k < CHANNELS; ++k) { mean[k] = points[pos + k]; }
	}

	// set to ensure the first computation
	atomicOr((int*) &shared_continueIteration, true);

	// shared_continueIteration is true if at least one thread per block must continue
	while (shared_continueIteration)
	{
		float centroid[CHANNELS];

		// initialize the centroid to 0 to accumulate points later
		for (float& k : centroid) { k = 0; }

		// track the number of points inside the const_squaredBandwidth window
		int windowPoints = 0;

		for (int phaseY = 0; phaseY < phasesY; ++phaseY)
		{
			for (int phaseX = 0; phaseX < phasesX; ++phaseX)
			{
				// check tile dimension against image boarder
				int tileDimX = min(TILE_WIDTH, width - TILE_WIDTH * phaseX);
				int tileDimY = min(TILE_WIDTH, height - TILE_WIDTH * phaseY);

				int loadingStepsX = std::ceil((float) TILE_WIDTH / THREADS_X);
				int loadingStepsY = std::ceil((float) TILE_WIDTH / THREADS_Y);

                for (int i=0; i<loadingStepsX; i++){
                    for(int j=0; j<loadingStepsY; j++){
                        if (ty + THREADS_Y * j < tileDimY && tx + THREADS_X * i < tileDimX)
                        {
                            unsigned int phaseRow = phaseY * TILE_WIDTH + row % TILE_WIDTH + THREADS_Y * j;
                            unsigned int phaseCol = phaseX * TILE_WIDTH + col % TILE_WIDTH + THREADS_X * i;
                            unsigned int phasePos = (phaseRow * width + phaseCol) * CHANNELS;

                            for (int k = 0; k < CHANNELS; ++k) { shared_tile[ty + THREADS_Y * j][(tx + THREADS_X * i) * CHANNELS + k] = points[phasePos + k]; }
                        }
                    }
                }

				__syncthreads();

				// compute the mean
				if (private_continueIteration && row < height && col < width)
				{
					for (int tileRow = 0; tileRow < tileDimY; ++tileRow)
					{
						for (int tileCol = 0; tileCol < tileDimX; ++tileCol)
						{
							float point[CHANNELS];
							for (int k = 0; k < CHANNELS; ++k) { point[k] = shared_tile[tileRow][tileCol * CHANNELS + k]; }

							if (l2SquaredDistance_cuda(mean, point, CHANNELS) <= const_squaredBandwidth)
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

                // reset shared_continueIteration
                atomicAnd((int*) &shared_continueIteration, false);

				__syncthreads();
			}
		}

		// check if the thread pixel is not outside the image
		if (private_continueIteration && row < height && col < width) {
			// get the centroid dividing by the number of points taken into account
			for (float& k : centroid) { k /= (float) windowPoints; }

			float shift = l2SquaredDistance_cuda(mean, centroid, CHANNELS);

			// update the mean
			for (int k = 0; k < CHANNELS; ++k) { mean[k] = centroid[k]; }

			private_continueIteration = false;

			// set if the thread must continue, hence the block
			if (shift >= epsilon) {
				atomicOr((int *) &shared_continueIteration, true);
				private_continueIteration = true;
			}

			for (int k = 0; k < CHANNELS; ++k) { means[pos + k] = mean[k]; }
		}

		__syncthreads();
	}
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

    auto start_time_cuda = std::chrono::high_resolution_clock::now();

    // Launch the kernel on the GPU
    matrixMeanShiftCUDA_kernel<<<gridSize, blockSize>>>(dev_points, dev_means, width, height);

    // Wait for the kernel to finish
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    auto end_time_cuda = std::chrono::high_resolution_clock::now();

    // Copy the result array from device to host memory
    auto* means = new float[nOfPoints * dimension];
    CUDA_CHECK_RETURN(cudaMemcpy(means, dev_means, nOfPoints * dimension * sizeof(float), cudaMemcpyDeviceToHost));

    // Free the GPU buffers
    CUDA_CHECK_RETURN(cudaFree(dev_points));
    CUDA_CHECK_RETURN(cudaFree(dev_means));

    // sequential phase

    auto start_time_sequential = std::chrono::high_resolution_clock::now();

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
            if (l2SquaredDistance(mean, mode, dimension) < squaredBandwidth)
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

    auto end_time_sequential = std::chrono::high_resolution_clock::now();

    // timings
    float totalTime_cuda = (float) std::chrono::duration_cast<std::chrono::microseconds>(end_time_cuda - start_time_cuda).count() / 1000.f;
    float totalTime_sequential = (float) std::chrono::duration_cast<std::chrono::microseconds>(end_time_sequential - start_time_sequential).count() / 1000.f;
    float totalTime = totalTime_cuda + totalTime_sequential;

    /*printf("Cuda timings:");
    printf("  cuda:   %fms\n", totalTime_cuda);
    printf("  sequential: %fms\n", totalTime_sequential);
    printf("  total: %fms\n", totalTime);*/

    return clustersCount;
}


