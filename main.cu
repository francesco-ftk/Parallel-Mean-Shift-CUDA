#include <iostream>
#include <chrono>
#include <filesystem>

#include "cpp_sources/ppm_io.cpp"
#include "cuda_sources/color_converter.cu"

#include "cuda_sources/matrix_meanshift_cuda.cu"

#define INPUT_FOLDER "../img/input/"
#define INPUT_PATH "../img/balloons_50.ppm"
#define OUTPUT_PATH "../img/out.ppm"
#define ITERATIONS 1
#define BANDWIDTH 0.4
#define COLOR_SPACE_DIMENSION 3
#define CLUSTERING_SPACE_DIMENSION 5
#define RGB_MAX_VALUE 255
#define HUE_MAX_VALUE 360

using namespace std::chrono;
namespace fs = std::filesystem;


int imageIteration(std::string inputPath) {
	// open the ppm image
	PPM ppm;
	if (ppm.read(inputPath) != 0)
	{
		std::cout << "ERROR: failed to open the image";
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

	}

	// create the index array
	int* clusters = new int[nOfPixels];

	// create the result variables
	int nOfClusters;
	float totalTime = 0;

	// function loop
	for (int i = 0; i < ITERATIONS; ++i)
	{
		//printf("Calling the MeanShift function... (%d)\n", i);

		// time the function
		auto start_time = high_resolution_clock::now();
		nOfClusters = matrixMeanShiftCUDA(pixels, BANDWIDTH, CLUSTERING_SPACE_DIMENSION, modes, clusters, width, height);
		auto end_time = high_resolution_clock::now();

		totalTime += duration_cast<microseconds>(end_time - start_time).count() / 1000.f;
	}

	float averageTime = totalTime / ITERATIONS;

	// print the results
	//printf("Matrix timings: (measured on %d iterations)\n", ITERATIONS);
	printf("CUDA timing (%s)\n", inputPath.c_str());
	printf("  total:   %fms\n", totalTime);
	//printf("  average: %fms\n", averageTime);
	printf("Number of clusters: %d\n", nOfClusters);

	printf("\n");

	auto* outputBuffer = new uint8_t[nOfPixels * COLOR_SPACE_DIMENSION];

	// populate the output image buffer
    for (int i = 0; i < nOfPixels; ++i)
	{
        float fR = modes[clusters[i] * CLUSTERING_SPACE_DIMENSION];
		float fG = modes[clusters[i] * CLUSTERING_SPACE_DIMENSION + 1];
		float fB = modes[clusters[i] * CLUSTERING_SPACE_DIMENSION + 2];


		int R = (int) (fR * RGB_MAX_VALUE);
		int G = (int) (fG * RGB_MAX_VALUE);
		int B = (int) (fB * RGB_MAX_VALUE);

		outputBuffer[i * COLOR_SPACE_DIMENSION]	    = R;
		outputBuffer[i * COLOR_SPACE_DIMENSION + 1] = G;
		outputBuffer[i * COLOR_SPACE_DIMENSION + 2] = B;

	}

	ppm.load(outputBuffer, height, width, ppm.getMax(), ppm.getMagic());

	// write the output ppm image
	if (ppm.write(OUTPUT_PATH) != 0)
	{
		std::cout << "ERROR: failed to write the image";
		return -1;
	}

	delete[] pixels;
	delete[] modes;
	delete[] clusters;
	delete[] outputBuffer;

	return 0;
}

int main() {
	for (const auto & b_entry : fs::directory_iterator(INPUT_FOLDER)) {
		imageIteration(b_entry.path().string());
	}
	return 0;
}

