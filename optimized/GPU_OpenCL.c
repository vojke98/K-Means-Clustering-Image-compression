#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "FreeImage.h"
#include <math.h>
#include <CL/cl.h>
#include <time.h>

#define WORKGROUP_SIZE 16
#define MAX_SOURCE_SIZE 16384

struct Point
{
    int r, g, b, a;
};

cl_int status;

int main(int argc, const char *argv[])
{

    srandom(time(NULL));
    ulong randomSeed = random();

    char imageName[100];
    char imageOutName[100];
    int numberOfClusters = 0;
    int numberOfIterations = 0;

    if (argc != 5)
    {
        printf("USAGE: ./GPU_OpenCL input_image output_image number_of_clusters number_of_iterations\n");
        exit(EXIT_SUCCESS);
    }

    sprintf(imageName, "%s", argv[1]);
    sprintf(imageOutName, "%s", argv[2]);
    numberOfClusters = atoi(argv[3]);
    numberOfIterations = atoi(argv[4]);

    // Load image from file
    FIBITMAP *imageBitmap = FreeImage_Load(FIF_PNG, imageName, PNG_DEFAULT);
    FIBITMAP *imageBitmap32 = FreeImage_ConvertTo32Bits(imageBitmap);

    // Get image dimensions
    int width = FreeImage_GetWidth(imageBitmap32);
    int height = FreeImage_GetHeight(imageBitmap32);
    int pitch = FreeImage_GetPitch(imageBitmap32);

    // Preapare room for a raw data copy of the image
    size_t imageSize = height * pitch * sizeof(char);

    unsigned char *image = malloc(imageSize);
    FreeImage_ConvertToRawBits(image, imageBitmap32, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

    FILE *fp = fopen("kernel.cl", "r");
    if (!fp)
    {
        fprintf(stderr, "Could not open kernel.\n");
        exit(1);
    }

    // Zapiši Kernel v RAM
    char *source_str = (char *)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    source_str[source_size] = '\0';
    fclose(fp);

    // Podatki o platformi
    cl_platform_id platform_id[10];
    cl_uint num_platforms;
    status = clGetPlatformIDs(10, platform_id, &num_platforms); // Max. število platform, kazalec na platforme, dejansko število platform

    // Podatki o napravi
    cl_device_id device_id[10];
    cl_uint num_devices;
    // Delali bomo s platform_id[0] na GPU
    status = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10, device_id, &num_devices);
    // izbrana platforma, tip naprave, koliko naprav nas zanima
    // kazalec na naprave, dejansko število naprav

    // Kontekst
    cl_context context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, &status);
    // kontekst: vkljucene platforme - NULL je privzeta, število naprav,
    // kazalci na naprave, kazalec na call-back funkcijo v primeru napake
    // dodatni parametri funkcije, številka napake

    // Ukazna vrsta
    cl_command_queue commandQueue = clCreateCommandQueue(context, device_id[0], 0, &status);
    // kontekst, naprava, INORDER/OUTOFORDER, napake

    // Delitev dela na podlagi velikosti vhodne slike
    const size_t localItemSize1 = 256;
    const size_t num_groups1 = (((width * height) - 1) / localItemSize1 + 1);
    const size_t globalItemSize1 = num_groups1 * localItemSize1;

    // Delitev dela na podlagi števila barv
    const size_t localItemSize2 = 16;
    const size_t num_groups2 = ((numberOfClusters - 1) / localItemSize2 + 1);
    const size_t globalItemSize2 = num_groups2 * localItemSize2;

    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Alokacija pomnilnika na napravi
    cl_mem image_d = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, imageSize, image, &status);
    cl_mem centroids_d = clCreateBuffer(context, CL_MEM_READ_WRITE, numberOfClusters * sizeof(struct Point), NULL, &status);
    cl_mem c_d = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * sizeof(int), NULL, &status);
    cl_mem sum_d = clCreateBuffer(context, CL_MEM_READ_WRITE, numberOfClusters * sizeof(struct Point), NULL, &status);
    cl_mem n_d = clCreateBuffer(context, CL_MEM_READ_WRITE, numberOfClusters * sizeof(int), NULL, &status);

    // Priprava programa
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, NULL, &status);
    // kontekst, število kazalcev na kodo, kazalci na kodo,
    // stringi so NULL terminated, napaka

    // Prevajanje
    status = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);
    // program, število naprav, lista naprav, opcije pri prevajanju,
    // kazalec na funkcijo, uporabniski argumenti

    // Log
    size_t build_log_len;
    char *build_log;
    status = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    // program, naprava, tip izpisa,
    // maksimalna dolžina niza, kazalec na niz, dejanska dolžina niza
    build_log = (char *)malloc(sizeof(char) * (build_log_len + 1));
    status = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, build_log_len, build_log, NULL);
    printf("%s\n", build_log);
    free(build_log);

    // Ščepec: priprava objekta
    cl_kernel initializeValues_kernel = clCreateKernel(program, "initialize_values", &status);
    cl_kernel arrangeInClusters_kernel = clCreateKernel(program, "arrange_in_clusters", &status);
    cl_kernel updateCentroidValues_kernel = clCreateKernel(program, "update_centroid_values", &status);
    cl_kernel rebuildImage_kernel = clCreateKernel(program, "rebuild_image", &status);

    // Ščepec: argumenti
    status = clSetKernelArg(initializeValues_kernel, 0, sizeof(cl_mem), (void *)&image_d);
    status |= clSetKernelArg(initializeValues_kernel, 1, sizeof(cl_int), (void *)&width);
    status |= clSetKernelArg(initializeValues_kernel, 2, sizeof(cl_int), (void *)&height);
    status |= clSetKernelArg(initializeValues_kernel, 3, sizeof(cl_mem), (void *)&centroids_d);
    status |= clSetKernelArg(initializeValues_kernel, 4, sizeof(cl_int), (void *)&numberOfClusters);
    status |= clSetKernelArg(initializeValues_kernel, 5, sizeof(ulong), (void *)&randomSeed);

    status |= clSetKernelArg(arrangeInClusters_kernel, 0, sizeof(cl_mem), (void *)&image_d);
    status |= clSetKernelArg(arrangeInClusters_kernel, 1, sizeof(cl_int), (void *)&width);
    status |= clSetKernelArg(arrangeInClusters_kernel, 2, sizeof(cl_int), (void *)&height);
    status |= clSetKernelArg(arrangeInClusters_kernel, 3, sizeof(cl_mem), (void *)&centroids_d);
    status |= clSetKernelArg(arrangeInClusters_kernel, 4, sizeof(cl_mem), (void *)&c_d);
    status |= clSetKernelArg(arrangeInClusters_kernel, 5, sizeof(cl_mem), (void *)&sum_d);
    status |= clSetKernelArg(arrangeInClusters_kernel, 6, sizeof(cl_mem), (void *)&n_d);
    status |= clSetKernelArg(arrangeInClusters_kernel, 7, sizeof(cl_int), (void *)&numberOfClusters);
    status |= clSetKernelArg(arrangeInClusters_kernel, 8, numberOfClusters * sizeof(struct Point), NULL);
    status |= clSetKernelArg(arrangeInClusters_kernel, 9, numberOfClusters * sizeof(int), NULL);

    status |= clSetKernelArg(updateCentroidValues_kernel, 0, sizeof(cl_mem), (void *)&image_d);
    status |= clSetKernelArg(updateCentroidValues_kernel, 1, sizeof(cl_int), (void *)&width);
    status |= clSetKernelArg(updateCentroidValues_kernel, 2, sizeof(cl_int), (void *)&height);
    status |= clSetKernelArg(updateCentroidValues_kernel, 3, sizeof(cl_mem), (void *)&centroids_d);
    status |= clSetKernelArg(updateCentroidValues_kernel, 4, sizeof(cl_mem), (void *)&sum_d);
    status |= clSetKernelArg(updateCentroidValues_kernel, 5, sizeof(cl_mem), (void *)&n_d);
    status |= clSetKernelArg(updateCentroidValues_kernel, 6, sizeof(cl_int), (void *)&numberOfClusters);
    status |= clSetKernelArg(updateCentroidValues_kernel, 7, sizeof(ulong), (void *)&randomSeed);

    status |= clSetKernelArg(rebuildImage_kernel, 0, sizeof(cl_mem), (void *)&image_d);
    status |= clSetKernelArg(rebuildImage_kernel, 1, sizeof(cl_int), (void *)&width);
    status |= clSetKernelArg(rebuildImage_kernel, 2, sizeof(cl_int), (void *)&height);
    status |= clSetKernelArg(rebuildImage_kernel, 3, sizeof(cl_mem), (void *)&centroids_d);
    status |= clSetKernelArg(rebuildImage_kernel, 4, sizeof(cl_mem), (void *)&c_d);
    // ščepec, številka argumenta, velikost podatkov, kazalec na podatke

    cl_event *events;
    // Ščepec: zagon
    status = clEnqueueNDRangeKernel(commandQueue, initializeValues_kernel, 1, NULL, &globalItemSize2, &localItemSize2, 0, NULL, NULL);
    // vrsta, ščepec, dimenzionalnost, mora biti NULL,
    // kazalec na število vseh niti, kazalec na lokalno število niti,
    // dogodki, ki se morajo zgoditi pred klicem

    for (size_t i = 0; i < numberOfIterations; i++)
    {
        status = clEnqueueNDRangeKernel(commandQueue, arrangeInClusters_kernel, 1, NULL, &globalItemSize1, &localItemSize1, 0, NULL, NULL);
        status = clEnqueueNDRangeKernel(commandQueue, updateCentroidValues_kernel, 1, NULL, &globalItemSize2, &localItemSize2, 0, NULL, NULL);
    }

    status = clEnqueueNDRangeKernel(commandQueue, rebuildImage_kernel, 1, NULL, &globalItemSize1, &localItemSize1, 0, NULL, NULL);

    // Čakanje na konec izvajanja vseh ščepcev
    // clWaitForEvents(numberOfIterations + 2, events);

    // Kopiranje rezultatov
    status = clEnqueueReadBuffer(commandQueue, image_d, CL_TRUE, 0, imageSize, image, 0, NULL, NULL);
    // branje v pomnilnik iz naprave, 0 = offset
    // zadnji trije dogodki, ki se morajo zgoditi prej

    // Izračun časa izvajanja
    clock_gettime(CLOCK_MONOTONIC, &finish);
    double elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("Čas izvajanja programa: %f sekund\n", elapsed);

    // Write output image to file
    FIBITMAP *imageOutBitmap32 = FreeImage_ConvertFromRawBits(image, width, height, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);
    FreeImage_Save(FIF_PNG, imageOutBitmap32, imageOutName, 0);

    // Cleanup
    status = clFlush(commandQueue);
    status |= clFinish(commandQueue);
    status |= clReleaseKernel(initializeValues_kernel);
    status |= clReleaseKernel(arrangeInClusters_kernel);
    status |= clReleaseKernel(updateCentroidValues_kernel);
    status |= clReleaseKernel(rebuildImage_kernel);
    status |= clReleaseProgram(program);
    status |= clReleaseMemObject(image_d);
    status |= clReleaseMemObject(centroids_d);
    status |= clReleaseMemObject(c_d);
    status |= clReleaseMemObject(sum_d);
    status |= clReleaseMemObject(n_d);
    status |= clReleaseCommandQueue(commandQueue);
    status |= clReleaseContext(context);

    // Free source image data
    FreeImage_Unload(imageBitmap32);
    FreeImage_Unload(imageOutBitmap32);
    FreeImage_Unload(imageBitmap);

    free(image);

    return 0;
}