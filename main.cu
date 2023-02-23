#include <iostream>
#include <iostream>
#include <chrono>
#include "cuda_runtime.h"

#include "cpp_sources/ppm_io.cpp"
//#include "cpp_sources/matrix_meanshift.cpp"
//#include "cpp_sources/soa_meanshift.cpp"
#include "cpp_sources/rgb_pixels.cpp"
#include "cuda_sources/color_converter.cu"

#include "cuda_sources/matrix_meanshift_cuda.cu"

#define INPUT_PATH "../img/image_bigger.ppm"
#define OUTPUT_PATH "../img/image_bigger_out_cuda_rgb.ppm"
#define ITERATIONS 1
#define BANDWIDTH 0.4
#define COLOR_SPACE_DIMENSION 3
#define CLUSTERING_SPACE_DIMENSION 5
#define RGB_MAX_VALUE 255
#define HUE_MAX_VALUE 360

/* ----- TIMINGS ------------------------------
 * 100x100 image, Windows, 12 cores, 18 threads, block 16x16, tile 16x16
 * 	 Matrix sequential: 3609ms	(release)
 * 	 Matrix sequential: ???		(debug)
 * 	 Matrix OpenMP:		1029ms	(release)
 * 	 Matrix OpenMP:		???		(debug)
 *   SoA sequential:	3834ms	(release)
 *   SoA sequential:	???		(debug)
 * 	 SoA OpenMP:		1060ms	(release)
 * 	 SoA OpenMP:		???		(debug)
 * 	 Matrix Cuda:		1973ms	(release)
 * 	 Matrix Cuda:		2012ms	(debug)
 *
 *	 Speedup OpenMP Matrix:		3.5 (release)
 * 	 Speedup OpenMP Matrix:		3.5 (debug)
 * 	 Speedup OpenMP SoA: 		3.6 (release)
 * 	 Speedup OpenMP SoA: 		3.6 (debug)
 * 	 Speedup Matrix Cuda: 		??? (release)
 * 	 Speedup Matrix Cuda: 		??? (debug)
 *
 * 100x100 image, Linux, 8 cores, 12 threads, block 16x16, tile 16x16
 *
 * 	 Matrix sequential: 2461ms	(release)
 * 	 Matrix sequential: ???		(debug)
 * 	 Matrix OpenMP:		998ms	(release)
 * 	 Matrix OpenMP:		???		(debug)
 *   SoA sequential:	2711ms	(release)
 *   SoA sequential:	???		(debug)
 * 	 SoA OpenMP:		726ms	(release)
 * 	 SoA OpenMP:		???		(debug)
 * 	 Matrix Cuda:		6945ms  (release)
 * 	 Matrix Cuda:		???		(debug)
 *
 *	 Speedup OpenMP Matrix:		2.5 (release)
 * 	 Speedup OpenMP Matrix:		??? (debug)
 * 	 Speedup OpenMP SoA: 		3.7 (release)
 * 	 Speedup OpenMP SoA: 		??? (debug)
 * 	 Speedup Matrix Cuda: 		??? (release)
 * 	 Speedup Matrix Cuda: 		??? (debug)
 *
 * Averaged on 10 iterations
 * --------------------------------------------
 */

// todo: cluster in the HSV space
// todo: cluster in the L*U*V* space
// todo: kernel multiplication
// todo: parallelize using Cuda

int main()
{
	// open the ppm image
	PPM ppm;
	if (ppm.read(INPUT_PATH) != 0)
	{
		cout << "ERROR: failed to open the image";
		return -1;
	}
	int width = ppm.getW();
	int height = ppm.getH();
	int nOfPixels = width * height;
	uint8_t* inputBuffer = ppm.getImageHandler();

	// create the matrices
	auto* pixels = new float[nOfPixels * CLUSTERING_SPACE_DIMENSION];
	auto* modes  = new float[nOfPixels * CLUSTERING_SPACE_DIMENSION];

	// initialize the pixel data
	for (int i = 0; i < nOfPixels; ++i)
	{
		int R = inputBuffer[i * COLOR_SPACE_DIMENSION];
		int G = inputBuffer[i * COLOR_SPACE_DIMENSION + 1];
		int B = inputBuffer[i * COLOR_SPACE_DIMENSION + 2];

		/*double X,Y,Z;
		RGBtoXYZ(R, G, B, X, Y, Z);
		Xmax = std::max(X, Xmax);
		Ymax = std::max(Y, Ymax);
		Zmax = std::max(Z, Zmax);*/

		float fR = (float) R / RGB_MAX_VALUE;
		float fG = (float) G / RGB_MAX_VALUE;
		float fB = (float) B / RGB_MAX_VALUE;

		div_t i_div_width = std::div(i, width);
		float fX = (float) i_div_width.rem / (float) (width - 1);
		float fY = (float) i_div_width.quot / (float) (height - 1);

		pixels[i * CLUSTERING_SPACE_DIMENSION]     = fR;
		pixels[i * CLUSTERING_SPACE_DIMENSION + 1] = fG;
		pixels[i * CLUSTERING_SPACE_DIMENSION + 2] = fB;
		pixels[i * CLUSTERING_SPACE_DIMENSION + 3] = fX;
		pixels[i * CLUSTERING_SPACE_DIMENSION + 4] = fY;

		/*if ((i + 500) % 1000 == 0) {
			printf("------------------------------\n");
			printf("R:\t%d\tG:\t%d\tB:\t%d\n", R, G, B);
			printf("X:\t%f\tY:\t%f\tZ:\t%f\n", X, Y, Z);
			//printf("fR:\t%f\tfG:\t%f\tfB:\t%f\n", fR, fG, fB);
			//printf("fH:\t%f\tfS:\t%f\tfV:\t%f\n", fH, fS, fV);
			printf("fX:\t%f\tfY:\t%f\n", fX, fY);
			printf("------------------------------\n");
		}*/
	}

 /*
    printf("Xmax:\t%f\n", Xmax);
	printf("Ymax:\t%f\n", Ymax);
	printf("Zmax:\t%f\n", Zmax);
 */

	// create the index array
	int* clusters = new int[nOfPixels];

	// create the result variables
	int nOfClusters;
	float totalTime = 0;

	// function loop
	for (int i = 0; i < ITERATIONS; ++i)
	{
		printf("Calling the MeanShift function... (%d)\n", i);

		// time the function
		auto start_time = chrono::high_resolution_clock::now();
		nOfClusters = matrixMeanShiftCUDA(pixels, BANDWIDTH, CLUSTERING_SPACE_DIMENSION, modes, clusters, width, height);
		auto end_time = chrono::high_resolution_clock::now();

		totalTime += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() / 1000.f;
	}

	float averageTime = totalTime / ITERATIONS;

	// print the results
	printf("Matrix timings: (measured on %d iterations)\n", ITERATIONS);
	printf("  total:   %fms\n", totalTime);
	printf("  average: %fms\n", averageTime);
	printf("Number of clusters: %d\n", nOfClusters);

	printf("\n");

	auto* outputBuffer = new uint8_t[nOfPixels * COLOR_SPACE_DIMENSION];

	// populate the output image buffer
    for (int i = 0; i < nOfPixels; ++i)
	{
        float fR = modes[clusters[i] * CLUSTERING_SPACE_DIMENSION];
		float fG = modes[clusters[i] * CLUSTERING_SPACE_DIMENSION + 1];
		float fB = modes[clusters[i] * CLUSTERING_SPACE_DIMENSION + 2];
		//float fX = modes[clusters[i] * CLUSTERING_SPACE_DIMENSION + 3];
		//float fY = modes[clusters[i] * CLUSTERING_SPACE_DIMENSION + 4];

		/*double R, G, B;
		XYZtoRGB(X, Y, Z, R, G, B);*/

		int R = (int) (fR * RGB_MAX_VALUE);
		int G = (int) (fG * RGB_MAX_VALUE);
		int B = (int) (fB * RGB_MAX_VALUE);

		outputBuffer[i * COLOR_SPACE_DIMENSION]	    = R;
		outputBuffer[i * COLOR_SPACE_DIMENSION + 1] = G;
		outputBuffer[i * COLOR_SPACE_DIMENSION + 2] = B;

		/*if ((i + 500) % 1000 == 0) {
			printf("------------------------------\n");
			printf("R:\t%d\tG:\t%d\tB:\t%d\n", R, G, B);
			printf("fR:\t%f\tfG:\t%f\tfB:\t%f\n", fR, fG, fB);
			printf("fH:\t%f\tfS:\t%f\tfV:\t%f\n", fH, fS, fV);
			printf("fX:\t%f\tfY:\t%f\n", fX, fY);
			printf("------------------------------\n");
		}*/
	}

    // printf("R: %f, G: %f, B: %f, H, %f, S: %f, V: %f \n", fR, fG, fB, fH, fS, fV);

	ppm.load(outputBuffer, height, width, ppm.getMax(), ppm.getMagic());

	// write the output ppm image
	if (ppm.write(OUTPUT_PATH) != 0)
	{
		cout << "ERROR: failed to write the image";
		return -1;
	}

	delete[] pixels;
	delete[] modes;
	delete[] clusters;
	delete[] outputBuffer;

	return 0;
}

