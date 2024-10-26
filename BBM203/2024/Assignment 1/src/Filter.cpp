#include "Filter.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <numeric>
#include <math.h>

void Filter::apply_mean_filter(GrayscaleImage& image, int kernelSize) {
    int width = image.get_width();
    int height = image.get_height();
    int halfKernel = kernelSize / 2;
    GrayscaleImage originalImage = image; 

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            int count = 0;
            
            for (int ky = -halfKernel; ky <= halfKernel; ++ky) {
                for (int kx = -halfKernel; kx <= halfKernel; ++kx) {
                    int nx = x + kx;
                    int ny = y + ky;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) { 
                        sum += originalImage.get_pixel(ny, nx);
                        ++count;
                    }
                }
            }
            image.set_pixel(y, x, sum / count);
        }
    }
}


void Filter::apply_gaussian_smoothing(GrayscaleImage& image, int kernelSize, double sigma) {
    int width = image.get_width();
    int height = image.get_height();
    int halfKernel = kernelSize / 2;

    std::vector<std::vector<double>> kernel(kernelSize, std::vector<double>(kernelSize));
    double sum = 0.0;

    for (int i = -halfKernel; i <= halfKernel; ++i) {
        for (int j = -halfKernel; j <= halfKernel; ++j) {
            double value = std::exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            kernel[i + halfKernel][j + halfKernel] = value;
            sum += value;
        }
    }

    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernel[i][j] /= sum;
        }
    }

    GrayscaleImage originalImage = image; 

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double weightedSum = 0.0;

            for (int ky = -halfKernel; ky <= halfKernel; ++ky) {
                for (int kx = -halfKernel; kx <= halfKernel; ++kx) {
                    int nx = x + kx;
                    int ny = y + ky;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        weightedSum += originalImage.get_pixel(ny, nx) * kernel[ky + halfKernel][kx + halfKernel];
                    }
                }
            }

            image.set_pixel(y, x, static_cast<int>(std::round(weightedSum)));
        }
    }
}

void Filter::apply_unsharp_mask(GrayscaleImage& image, int kernelSize, double amount) {
    int width = image.get_width();
    int height = image.get_height();
    GrayscaleImage blurredImage = image; 

    apply_gaussian_smoothing(blurredImage, kernelSize, 1.0); 

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int original = image.get_pixel(y, x);
            int blurred = blurredImage.get_pixel(y, x);

            int sharpened = static_cast<int>(original + amount * (original - blurred));
            sharpened = std::max(0, std::min(255, sharpened)); 

            image.set_pixel(y, x, sharpened);
        }
    }
}
