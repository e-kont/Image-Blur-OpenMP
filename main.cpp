#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <chrono>
#include <omp.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int KERNEL_RADIUS = 8;
const float sigma = 3.f;

unsigned char blurAxis(int x, int y, int channel, int axis/*0: horizontal axis, 1: vertical axis*/, unsigned char* input, int width, int height)
{
	float sum_weight = 0.0f;
	float ret = 0.f;

	for (int offset = -KERNEL_RADIUS; offset <= KERNEL_RADIUS; offset++)
	{
		int offset_x = axis == 0 ? offset : 0;
		int offset_y = axis == 1 ? offset : 0;
		int pixel_y = std::max(std::min(y + offset_y, height - 1), 0);
		int pixel_x = std::max(std::min(x + offset_x, width - 1), 0);
		int pixel = pixel_y * width + pixel_x;

		float weight = std::exp(-(offset * offset) / (2.f * sigma * sigma));

		ret += weight * input[4 * pixel + channel];
		sum_weight += weight;
	}
	ret /= sum_weight;

	return (unsigned char)std::max(std::min(ret, 255.f), 0.f);
}

void gaussian_blur_separate_parallel(const char* filename) {
	int width = 0;
	int height = 0;
	int img_orig_channels = 4;
	// Load image
	unsigned char* img_in = stbi_load(filename, &width, &height, &img_orig_channels, 4);
	if (img_in == nullptr)
	{
		printf("Could not load %s\n", filename);
		return;
	}

	unsigned char* img_out = new unsigned char[width * height * 4];
	unsigned char* img_horizontal_blur = new unsigned char[width * height * 4];

	// Start timer
	auto start = std::chrono::high_resolution_clock::now();

	#pragma omp parallel
	{
		// Horizontal blur
		#pragma omp for
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int pixel = y * width + x;
				for (int channel = 0; channel < 4; channel++)
				{
					img_horizontal_blur[4 * pixel + channel] = blurAxis(x, y, channel, 0, img_in, width, height);
				}
			}
		}

		// Vertical blur
		#pragma omp for
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int pixel = y * width + x;
				for (int channel = 0; channel < 4; channel++)
				{
					img_out[4 * pixel + channel] = blurAxis(x, y, channel, 1, img_horizontal_blur, width, height);
				}
			}
		}
	}

	// End timer
	auto end = std::chrono::high_resolution_clock::now();
	// Computation time in milliseconds
	int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Gaussian Blur Separate - Parallel: Time %dms\n", time);

	// Write final blurred image to file
	stbi_write_jpg("blurred_image_parallel.jpg", width, height, 4, img_out, 90);
	stbi_image_free(img_in);
	delete[] img_horizontal_blur;
	delete[] img_out;
}

void bloom_parallel(const char* filename) {
	int width = 0;
	int height = 0;
	int img_orig_channels = 4;
	// Load image
	unsigned char* img_in = stbi_load(filename, &width, &height, &img_orig_channels, 4);
	if (img_in == nullptr)
	{
		printf("Could not load %s\n", filename);
		return;
	}

	unsigned char* bloom_mask = new unsigned char[width * height * 4];
	unsigned char* bloom_blur_horizontal = new unsigned char[width * height * 4];
	unsigned char* bloom_blur = new unsigned char[width * height * 4];
	unsigned char* img_out = new unsigned char[width * height * 4];
	unsigned char maxLuminance = 0;

	// Start timer
	auto start = std::chrono::high_resolution_clock::now();

	#pragma omp parallel
	{
		// Calculate maximum luminance
		#pragma omp for
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int pixel = y * width + x;
				// Calculate pixel luminance
				unsigned char luminance = 0;
				for (int channel = 0; channel < 3; channel++) {
					luminance += img_in[4 * pixel + channel];
				}
				luminance /= 3;
				if (luminance > maxLuminance) maxLuminance = luminance;
			}
		}

		#pragma omp master
		{
			printf("Maximum pixel luminance: %d\n", maxLuminance);
		}

		// Create bloom_mask
		#pragma omp for
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int pixel = y * width + x;
				// Calculate pixel luminance
				unsigned char luminance = 0;
				for (int channel = 0; channel < 3; channel++) {
					luminance += img_in[4 * pixel + channel];
				}
				luminance /= 3;
				for (int channel = 0; channel < 4; channel++) {
					bloom_mask[4 * pixel + channel] = luminance > 0.9 * maxLuminance ? img_in[4 * pixel + channel] : 0;
				}
			}
		}
		
		// Gaussian Blur Horizontal
		#pragma omp for
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int pixel = y * width + x;
				for (int channel = 0; channel < 4; channel++) {
					bloom_blur_horizontal[4 * pixel + channel] = blurAxis(x, y, channel, 0, bloom_mask, width, height);
				}
			}
		}

		// Gaussian Blur Vertical
		#pragma omp for
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int pixel = y * width + x;
				for (int channel = 0; channel < 4; channel++) {
					bloom_blur[4 * pixel + channel] = blurAxis(x, y, channel, 1, bloom_blur_horizontal, width, height);
				}
			}
		}

		// Write blurred image to file in one thread
		#pragma omp master
		{
			stbi_write_jpg("bloom_blurred.jpg", width, height, 4, bloom_blur, 90);
		}

		// Create final image
		#pragma omp for
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int pixel = y * width + x;
				for (int channel = 0; channel < 4; channel++) {
					img_out[4 * pixel + channel] = img_in[4 * pixel + channel] + bloom_blur[4 * pixel + channel];
				}
			}
		}

		// Write final image to file in one thread
		#pragma omp single
		{
			stbi_write_jpg("bloom_final.jpg", width, height, 4, img_out, 90);
		}
	}

	// End timer
	auto end = std::chrono::high_resolution_clock::now();
	// Computation time in milliseconds
	int time = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	printf("Bloom - Parallel: Time %dms\n", time);

	stbi_image_free(img_in);
	delete[] bloom_mask;
	delete[] bloom_blur_horizontal;
	delete[] bloom_blur;
	delete[] img_out;
}

int main() {
	const char* filename = "street_night.jpg";
	gaussian_blur_separate_parallel(filename);
	bloom_parallel(filename);
}