#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "FreeImage.h"
#include <time.h>
#include <math.h>


struct Point { int r, g, b, a; };


void calculate_centroid_mean(unsigned char *image, int width, int height, int index, unsigned char *centroids, int *nearestCentroids, int clusters);
void kmeans_sequential(unsigned char *imageIn, unsigned char *imageOut, int width, int height, int pitch, int clusters, int iterations);
void initialize_centroids(unsigned char *image, int width, int height, unsigned char *centroids, int clusters);
void build_image(unsigned char *image, int width, int height, unsigned char *centroids, int *nearestCentroids);
int get_nearest_centroid(unsigned char *centroids, int clusters, struct Point pointA);
float euclidean_distance(struct Point pointA, struct Point pointB);
int random_integer(int min, int max);


int main(int argc, char *argv[]) {
    char imageInName[100];
    char imageOutName[100];
    int clusters = 0;
    int iterations = 0;

    if (argc != 5) {
        printf("USAGE: ./KMeansClustering input_image output_image number_of_clusters number_of_iterations\n");
        //exit(EXIT_FAILURE);
        exit(EXIT_SUCCESS);
    }
 
    sprintf(imageInName, "%s", argv[1]);
    sprintf(imageOutName, "%s", argv[2]);
    clusters = atoi(argv[3]);
    iterations = atoi(argv[4]);

    FIBITMAP *imageInBitmap = FreeImage_Load(FIF_PNG, imageInName, PNG_DEFAULT);
    FIBITMAP *imageInBitmap32 = FreeImage_ConvertTo32Bits(imageInBitmap);

    int width = FreeImage_GetWidth(imageInBitmap32);
	int height = FreeImage_GetHeight(imageInBitmap32);
	int pitch = FreeImage_GetPitch(imageInBitmap32);
    int bpp = FreeImage_GetBPP(imageInBitmap32);

    printf("Image info: Width=%d; \tHeight=%d; \tPitch=%d; \tBPP=%d\n", width, height, pitch, bpp);

    unsigned char *imageIn = (unsigned char *)malloc(height * pitch * sizeof(unsigned char));
    unsigned char *imageOut = (unsigned char *)malloc(height * pitch * sizeof(unsigned char));
    FreeImage_ConvertToRawBits(imageIn, imageInBitmap32, pitch, bpp, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // image compression using k-means clustering algorithm
    kmeans_sequential(imageIn, imageOut, width, height, pitch, clusters, iterations);

    clock_gettime(CLOCK_MONOTONIC, &finish);
    double elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("Čas izvajanja programa: %f sekund\n", elapsed);

    FIBITMAP *imageOutBitmap32 = FreeImage_ConvertFromRawBits(imageOut, width, height, pitch, bpp, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);
	FreeImage_Save(FIF_PNG, imageOutBitmap32, imageOutName, 0);

    FreeImage_Unload(imageInBitmap);
    FreeImage_Unload(imageInBitmap32);
    FreeImage_Unload(imageOutBitmap32);
    free(imageIn);
    free(imageOut);
}


/**
 *   @brief Compresses the image using the k-mean algorithm
 *
 *   @param imageIn sample array
 *   @param imageOut output array
 *   @param width image width
 *   @param height image height
 *   @param pitch image pitch
 *   @param clusters number of clusters in centrodis
 *   @param iterations number of iterations for the algorithm
 */
void kmeans_sequential(unsigned char *imageIn, unsigned char *imageOut, int width, int height, int pitch, int clusters, int iterations) {
    // Inicializiraj začetne vrednosti centroidov µ1, µ2, . . . , µk tako, da jim prirediš naključen vzorec xi
    unsigned char *centroids = malloc(clusters * 4 * sizeof(char));
    initialize_centroids(imageIn, width, height, centroids, clusters);

    int *nearestCentroids = malloc(width * height * sizeof(int));

    for(int i = 0; i < iterations; i++) {
        // for each sample xi
        for(int sampleIndex = 0; sampleIndex < (width * height); sampleIndex++) {
            int baseIndex = sampleIndex * 4;
            struct Point sample = {imageIn[baseIndex + 0], imageIn[baseIndex + 1], imageIn[baseIndex + 2], imageIn[baseIndex + 3]};
                        
            // find the centroid nearest to the sample and store it at index that correspond to sample index
            nearestCentroids[sampleIndex] = get_nearest_centroid(centroids, clusters, sample);
        }

        // calculate mean for each cluster
        for(int centroidIndex = 0; centroidIndex < clusters; centroidIndex++) {
            calculate_centroid_mean(imageIn, width, height, centroidIndex, centroids, nearestCentroids, clusters);
        }
    }

    build_image(imageOut, width, height, centroids, nearestCentroids);

    free(nearestCentroids);
    free(centroids);
}


/**
 *   @brief Sets values of given centroids array to the random sample from given image
 *
 *   @param image sample array
 *   @param width image width
 *   @param height image height
 *   @param centrodis centroid array
 *   @param clusters number of clusters in centrodis
 */
void initialize_centroids(unsigned char *image, int width, int height, unsigned char *centroids, int clusters) {
    int clusterIndex = 0;
    int max = width * height;
    int min = 0;

    for(int i = 0; i < (clusters * 4); i += 4){
        int r = random_integer(min, max) * 4;

        for(int c = 0; c < 4; c++) {
            centroids[i + c] = image[r + c];
        }
    }
}


/**
 *   @brief Returns the index of the nearest centroid relative to a given point
 *
 *   @param centroids array of centroids
 *   @param clusters number of clusters in centrodis
 *   @param pointA one structure Point
 *
 *   @return Index of the nearest centroid from centroids array relative to pointA
 */
int get_nearest_centroid(unsigned char *centroids, int clusters, struct Point pointA) {
    int index = 0;

    struct Point pointB = {centroids[0], centroids[1], centroids[2], centroids[3]};

    float min_deviation = euclidean_distance(pointA, pointB);

    for(int i = 4; i < (clusters * 4); i += 4){
        pointB.r = centroids[i + 0];
        pointB.g = centroids[i + 1];
        pointB.b = centroids[i + 2];
        pointB.a = centroids[i + 3];

        // distance represents the deviation between the colors of two points
        float deviation = euclidean_distance(pointA, pointB);

        if(deviation < min_deviation){
            index = i / 4;
            min_deviation = deviation;
        }
    }

    return index;
}


/**
 *   @brief Updates centroid at given index by calculating its mean value
 *
 *   @param image sample array
 *   @param width image width
 *   @param height image height
 *   @param index centroid base index
 *   @param centroids array of centroids
 *   @param nearestCentroids array that holds indexes of centroids that are nearest to the corresponding sample
 *   @param clusters number of clusters in centrodis
 */
void calculate_centroid_mean(unsigned char *image, int width, int height, int index, unsigned char *centroids, int *nearestCentroids, int clusters) {
    int numberOfPoints = 0;
    int baseIndex;

    struct Point point = {0, 0, 0, 0};

    for(int i = 0; i < (width * height); i++) {
        if(nearestCentroids[i] == index) {
            baseIndex = i * 4;
            
            point.r += image[baseIndex + 0];
            point.g += image[baseIndex + 1];
            point.b += image[baseIndex + 2];
            point.a += image[baseIndex + 3];

            numberOfPoints++;
        }
    }
    baseIndex = index * 4;

    // if cluster is empty, add random sample (real solution would be to add furthest sample)
    if(!numberOfPoints) {
        int max = width * height;
        int min = 0;
        int r = random_integer(min, max) * 4;

        for(int c = 0; c < 4; c++) {
            centroids[baseIndex + c] = image[r + c];
        }
    } else {
        centroids[baseIndex + 0] = point.r / numberOfPoints;
        centroids[baseIndex + 1] = point.g / numberOfPoints;
        centroids[baseIndex + 2] = point.b / numberOfPoints;
        centroids[baseIndex + 3] = point.a / numberOfPoints;
    }
}


/**
 *   @brief Creates new image data from given centroids
 *
 *   @param image sample array
 *   @param width image width
 *   @param height image height
 *   @param centroids array of centroids
 *   @param nearestCentroids array that holds indexes of centroids that are nearest to the corresponding sample
 */
void build_image(unsigned char *image, int width, int height, unsigned char *centroids, int *nearestCentroids) {
    for(int i = 0; i < (width * height * 4); i += 4) {
        int nearestCentroidBaseIndex = nearestCentroids[i / 4] * 4;

        for(int c = 0; c < 4; c++) {
            image[i + c] = centroids[nearestCentroidBaseIndex + c];
        }
    }
}


/**
 *   @brief Returns the Euclidean distance between its two input Points
 *
 *   @param pointA structure Point with 4 coorinate values
 *   @param pointB the other structure Point with 4 coorinate values
 *
 *   @return Euclidean distance between pointA and pointB 
 */
float euclidean_distance(struct Point pointA, struct Point pointB) {
    return pow(pointB.r - pointA.r, 2.0) + pow(pointB.g - pointA.g, 2.0) + pow(pointB.b - pointA.b, 2.0) + pow(pointB.a - pointA.a, 2.0);
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