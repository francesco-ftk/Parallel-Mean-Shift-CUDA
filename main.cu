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
#define OUTPUT_PATH "../img/image_bigger_out_cuda_hsv.ppm"
#define ITERATIONS 1
#define BANDWIDTH 0.4

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
 * 	 Matrix Cuda:		5219ms	(release)
 * 	 Matrix Cuda:		???		(debug)
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

using namespace std;
using namespace chrono;

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
	int rgbPixelSize = RgbPixels::COLOR_SPACE_DIMENSION;
	int rgbxySpaceSize = RgbPixels::SPACE_DIMENSION;
	int rgbMaxValue = RgbPixels::MAX_VALUE;
	float* pixels = new float[nOfPixels * rgbxySpaceSize];
	float* modes = new float[nOfPixels * rgbxySpaceSize];

    float fH;
    float fS;
    float fV;
    float fR;
    float fG;
    float fB;

	// initialize the pixel data
	for (int i = 0; i < nOfPixels; ++i)
	{
        fR= (float) inputBuffer[i * rgbPixelSize]/rgbMaxValue;
        fG= (float) inputBuffer[i * rgbPixelSize + 1]/rgbMaxValue;
        fB= (float) inputBuffer[i * rgbPixelSize + 2]/rgbMaxValue;
        RGBtoHSV(fR, fG, fB, fH, fS, fV);
		pixels[i * rgbxySpaceSize]     = (float) fH/360;                        // H
		pixels[i * rgbxySpaceSize + 1] = fS;                                    // S
		pixels[i * rgbxySpaceSize + 2] = fV;                                    // V
		pixels[i * rgbxySpaceSize + 3] = (float) ((i) % width) / (width - 1);	// X
		pixels[i * rgbxySpaceSize + 4] = (float) ((i) / width) / (height - 1);	// Y
	}

    // printf("R: %f, G: %f, B: %f, H, %f, S: %f, V: %f \n", fR, fG, fB, fH, fS, fV);

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
		auto start_time = high_resolution_clock::now();
		nOfClusters = matrixMeanShiftCUDA(pixels, BANDWIDTH, rgbxySpaceSize, modes, clusters, width, height);
		auto end_time = high_resolution_clock::now();

		totalTime += duration_cast<microseconds>(end_time - start_time).count() / 1000.f;
	}

	float averageTime = totalTime / ITERATIONS;

	// print the results
	printf("Matrix timings: (measured on %d iterations)\n", ITERATIONS);
	printf("  total:   %fms\n", totalTime);
	printf("  average: %fms\n", averageTime);
	printf("Number of clusters: %d\n", nOfClusters);

	printf("\n");

	// create the output image buffer
	rgbPixelSize = RgbPixels::COLOR_SPACE_DIMENSION;
	rgbMaxValue = RgbPixels::MAX_VALUE;
	uint8_t* outputBuffer = new uint8_t[nOfPixels * rgbPixelSize];
    for (int i = 0; i < nOfPixels; ++i)
	{
        fH=modes[clusters[i] * rgbxySpaceSize] * 360;
        fS=modes[clusters[i] * rgbxySpaceSize + 1];
        fV=modes[clusters[i] * rgbxySpaceSize + 2];
        HSVtoRGB(fR, fG, fB, fH, fS, fV);
		outputBuffer[i * rgbPixelSize]	   = (uint8_t) (fR * rgbMaxValue); // R
		outputBuffer[i * rgbPixelSize + 1] = (uint8_t) (fG * rgbMaxValue); // G
		outputBuffer[i * rgbPixelSize + 2] = (uint8_t) (fB * rgbMaxValue); // B
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

