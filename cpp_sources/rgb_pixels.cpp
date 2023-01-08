#ifndef RGB_PIXELS_CPP
#define RGB_PIXELS_CPP

// structure of arrays
struct RgbPixels {
	static const int COLOR_SPACE_DIMENSION = 3;
	static const int SPACE_DIMENSION = 5;
	static const int MAX_VALUE = 255;

	int width;
	int height;
	float* r;
	float* g;
	float* b;
	float* x;
	float* y;
	// todo: padding to better fit the cache

	void create(int _width, int _height) {
		width = _width;
		height = _height;
		r = new float[width * height];
		g = new float[width * height];
		b = new float[width * height];
		x = new float[width * height];
		y = new float[width * height];
	}

	void destroy() {
		delete[] r;
		delete[] g;
		delete[] b;
		delete[] x;
		delete[] y;
	}

	void load(uint8_t* buffer_image) {
		int j = 0;
		for (int i = 0; i < width * height * 3; i++) {
			if (i % 3 == 0) {
				r[j] = (float) buffer_image[i] / 255;
			} else if (i % 3 == 1) {
				g[j] = (float) buffer_image[i] / 255;
			} else {
				b[j] = (float) buffer_image[i] / 255;
				x[j] = (float) ((i / 3) % width) / (width - 1);
				y[j] = (float) ((i / 3) / width) / (height - 1);
				j++;
			}
		}
	}

	// write the elements at position i in the array
	void write(int i, float* array) {
		array[0] = r[i];
		array[1] = g[i];
		array[2] = b[i];
		array[3] = x[i];
		array[4] = y[i];
	}

	// saves the array values at the position i
	void save(float* array, int i) {
		r[i] = array[0];
		g[i] = array[1];
		b[i] = array[2];
		x[i] = array[3];
		y[i] = array[4];
	}

	void print(int i) {
		cout << "[ " << r[i] << " " << g[i] << " " << b[i] << " " << x[i] << " " << y[i] << " ]";
	}
};

#endif // RGB_PIXELS_CPP