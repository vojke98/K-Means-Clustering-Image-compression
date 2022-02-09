#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "FreeImage.h"


int random_integer(int min, int max);
void kmeans_sequential(unsigned char *imageIn, int width, int height, int numberOfClusters, int numberOfIterations);


int main(int argc, char *argv[]) {
    char imageInName[100];
    char imageOutName[100];
    int numberOfClusters = 0;
    int numberOfIterations = 0;

    if (argc != 5) {
        printf("USAGE: ./CPU_Sequential input_image output_image number_of_clusters number_of_iterations\n");
        exit(EXIT_SUCCESS);
    }

    sprintf(imageInName, "%s", argv[1]);
    sprintf(imageOutName, "%s", argv[2]);
    numberOfClusters = atoi(argv[3]);
    numberOfIterations = atoi(argv[4]);

    time_t t;
    srand((unsigned) time(&t));

	FIBITMAP *imageBitmap = FreeImage_Load(FIF_PNG, imageInName, PNG_DEFAULT);
    FIBITMAP *imageBitmap32 = FreeImage_ConvertTo32Bits(imageBitmap);

    int width = FreeImage_GetWidth(imageBitmap32);
    int height = FreeImage_GetHeight(imageBitmap32);
    int pitch = FreeImage_GetPitch(imageBitmap32);

    unsigned char *image = (unsigned char *)malloc(height * pitch * sizeof(unsigned char));
	FreeImage_ConvertToRawBits(image, imageBitmap32, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Image compression using k-means clustering algorithm
    kmeans_sequential(image, width, height, numberOfClusters, numberOfIterations);

    clock_gettime(CLOCK_MONOTONIC, &finish);
    double elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("Čas izvajanja programa: %f sekund\n", elapsed);

    // Save output image
    FIBITMAP *dst = FreeImage_ConvertFromRawBits(image, width, height, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);
	FreeImage_Save(FIF_PNG, dst, imageOutName, 0);

    // Cleanup
    free(image);

    return 0;
}


void kmeans_sequential(unsigned char *image, int width, int height, int numberOfClusters, int numberOfIterations) {
    unsigned char *centroids = malloc(numberOfClusters * 4 * sizeof(char)); // Array of centroids
    int *c = malloc(width * height * sizeof(int));                        // Array to store indexes of centroids nearest to corresponding samples
    int *sum = malloc(numberOfClusters * 4 * sizeof(int));                  // Array to store sum of RGBA values for each cluster
    int *n = malloc(numberOfClusters * sizeof(int));                        // Array to store number of elements in each cluster

    // Initialize values
    for (size_t i = 0; i < numberOfClusters * 4; i += 4) {
        int max = width * height;
        int min = 0;
        int r = random_integer(min, max) * 4;
        
        // Set centroid value to random sample
        centroids[i + 0] = image[r + 0];
        centroids[i + 1] = image[r + 1];
        centroids[i + 2] = image[r + 2];
        centroids[i + 3] = image[r + 3];

        // Set cluster sum to zero
        sum[i + 0] = 0;
        sum[i + 1] = 0;
        sum[i + 2] = 0;
        sum[i + 3] = 0;

        // Set number of elements to zero
        n[i / 4] = 0;
    }

    for (size_t i = 0; i < numberOfIterations; i++) {
        // For each sample, find the nearest centroid and assing it to the corresponding cluster
        for (size_t j = 0; j < (width * height); j++) {
            int nearestCentroidIndex = 0;
            int base = j * 4;

            unsigned char i_r = image[base + 0];
            unsigned char i_g = image[base + 1];
            unsigned char i_b = image[base + 2];
            unsigned char i_a = image[base + 3];

            unsigned char c_r = centroids[0];
            unsigned char c_g = centroids[1];
            unsigned char c_b = centroids[2];
            unsigned char c_a = centroids[3];

            // Set minimal deviation as distance between first two samples
            int minDeviation = pow(i_r - c_r, 2.0) + pow(i_g - c_g, 2.0) + pow(i_b - c_b, 2.0) + pow(i_a - c_a, 2.0);

            // Loop through centroids
            for (size_t k = 4; k < numberOfClusters * 4; k += 4) {
                c_r = centroids[k + 0];
                c_g = centroids[k + 1];
                c_b = centroids[k + 2];
                c_a = centroids[k + 3];
                
                // Find eucledian distance between two samples (deviation between two colors)
                int deviation = pow(i_r - c_r, 2.0) + pow(i_g - c_g, 2.0) + pow(i_b - c_b, 2.0) + pow(i_a - c_a, 2.0);

                // Update minimal deviation and index of second sample if new deviation is smaller than minimal deviation
                if (deviation < minDeviation) {
                    minDeviation = deviation;
                    nearestCentroidIndex = k;
                }
            }

            // At this point we have found cetroid nearest to pointA, so we store its index at corresponding position
            c[j] = nearestCentroidIndex;

            // Because we added one more sample to the cluster, we need to add it's RGBA values to the existing sum
            sum[nearestCentroidIndex + 0] += i_r;
            sum[nearestCentroidIndex + 1] += i_g;
            sum[nearestCentroidIndex + 2] += i_b;
            sum[nearestCentroidIndex + 3] += i_a;

            // New element is added to cluster, so we increase the number of elements in that specific cluster (nearestCentroidIndex)
            n[nearestCentroidIndex / 4]++;
        }

        // Loop through centroids to calculate average sample value
        for (size_t j = 0; j < numberOfClusters * 4; j += 4) {
            // centroids array is 4 times longer than n, so we need to normalize index
            int normalizedIndex = j / 4;

            // If there is no elements in the cluster, we append one random sample
            if (n[normalizedIndex] == 0) {
                int max = width * height;
                int min = 0;
                int r = random_integer(min, max) * 4;

                sum[j + 0] = image[r + 0];
                sum[j + 1] = image[r + 1];
                sum[j + 2] = image[r + 2];
                sum[j + 3] = image[r + 3];
                n[normalizedIndex]++;
            }

            // Set centroid RGBA values by dividing it's sum by corresponding number of elements inside cluster
            centroids[j + 0] = sum[j + 0] / n[normalizedIndex];
            centroids[j + 1] = sum[j + 1] / n[normalizedIndex];
            centroids[j + 2] = sum[j + 2] / n[normalizedIndex];
            centroids[j + 3] = sum[j + 3] / n[normalizedIndex];
        }
    }

    // Rebuild image using centroid data
    for (size_t i = 0; i < (width * height); i++) {
        // Index of centroid nearest to current point i
        int nearestCentroidIndex = c[i];
        unsigned char r = centroids[nearestCentroidIndex + 0];
        unsigned char g = centroids[nearestCentroidIndex + 1];
        unsigned char b = centroids[nearestCentroidIndex + 2];
        unsigned char a = centroids[nearestCentroidIndex + 3];

        // Image has 4 color channels, so we need to normalize current index by multiplying i by 4
        int imagePointIndex = i * 4;
        image[imagePointIndex + 0] = r;
        image[imagePointIndex + 1] = g;
        image[imagePointIndex + 2] = b;
        image[imagePointIndex + 3] = a;
    }
    
    // Cleanup
    free(centroids);
    free(c);
    free(sum);
    free(n);
}


/**
 *   @brief Returns the random integer in given range
 *
 *   @param min one integer value
 *   @param max the other integer value
 *
 *   @return Random integer that is greather or equal to min and smaller than max value 
 */
int random_integer(int min, int max) {
    return (rand() % (max - min)) + min;
}