# Image-Blur-OpenMP
Parallel image blurring using Gaussian blur filter with OpenMP.

# Included files

## street_night.jpg
Initial image expected to be blurred with gaussian_blur_separate_parallel() and bloom_parallel() functions.

## stb_image.h, stb_image_write.h
C++ libraries for reading and writing images.

## main.cpp
Main code to be executed.

# Functions

## gaussian_blur_separate_parallel()
Loads input image into pixel array (width * height * channels) and performs Gaussian blur on pixels in two axes, horizontal and vertical, using OpenMP for-loop parallelization. Result is written into a new image file named "blurred_image_parallel.jpg" by a single thread.

## bloom_parallel()
1. Loads input image into pixel array (width * height * channels)
2. Applies "bloom" filter on image: turns all image pixels dimmer than maximum pixel luminance into black color.
3. Performs Gaussian blur in two axes on image. Resulting image is written into a new image file named "bloom_blurred.jpg".
4. Merges original image and image with bloom filter applied into a new image. Result is written into a new image file named "bloom_final.jpg".

Pixels are accessed and processed using OpenMP for-loop parallelization, while output files are written by a single thread using #pragma omp master.
