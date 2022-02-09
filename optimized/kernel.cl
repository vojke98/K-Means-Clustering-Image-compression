struct Point {
    int r, g, b, a;
};

int euclidean_distance(struct Point pointA, struct Point pointB);
int random_integer(ulong seed, int min, int max);

__kernel void initialize_values(__global unsigned char *image, 
                                int width,
                                int height,
                                __global struct Point *centroids,
                                int numberOfClusters,
                                ulong randoms)
{
    int globalID = get_global_id(0);

    if(globalID < numberOfClusters) {
        int min = 0;
        int max = width * height;
        ulong seed = randoms + globalID;
        
        int r = random_integer(seed, min, max) * 4;
        
        // Set centroid value to random sample
        struct Point point = {image[r + 2], image[r + 1], image[r + 0], image[r + 3]};
        centroids[globalID] = point;
    }
}

__kernel void arrange_in_clusters(__global unsigned char *image, 
                                int width,
                                int height,
                                __global struct Point *centroids,
                                __global int *c,
                                __global struct Point *globalSum,
                                __global int *globalN,
                                int numberOfClusters,
                                __local struct Point *localSum,
                                __local int *localN)
{
    int globalID = get_global_id(0);

    if(globalID < width * height) {
        int localID = get_local_id(0);

        // Initialize local variables
        if(localID < numberOfClusters) {
            struct Point point = {0, 0, 0, 0};
            localSum[localID] = point;
            localN[localID] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Find nearest centroid
        int base = globalID * 4;
        struct Point pointA = {image[base + 2], image[base + 1], image[base + 0], image[base + 3]};
        struct Point pointB = centroids[0];

        int minDeviation = euclidean_distance(pointA, pointB);
        int nearestCentroidIndex = 0;
        
        // Loop through centroids
        for(int k = 1; k < numberOfClusters; k++) {
            pointB = centroids[k];

            // Find eucledian distance between two samples (deviation between two colors)
            int deviation = euclidean_distance(pointA, pointB);

            // Update minimal deviation and index of second sample if new deviation is smaller than minimal deviation
            if(deviation < minDeviation) {
                minDeviation = deviation;
                nearestCentroidIndex = k;
            }
        }
        // At this point we have found cetroid nearest to pointA, so we store its index at corresponding position
        c[globalID] = nearestCentroidIndex;

        // Because we added one more sample to the cluster, we need to add it's RGBA values to the existing sum
        atomic_add(&localSum[nearestCentroidIndex].r, pointA.r);
        atomic_add(&localSum[nearestCentroidIndex].g, pointA.g);
        atomic_add(&localSum[nearestCentroidIndex].b, pointA.b);
        atomic_add(&localSum[nearestCentroidIndex].a, pointA.a);

        // New element is added to cluster, so we increase the number of elements in that specific cluster (nearestCentroidIndex)
        atomic_inc(&localN[nearestCentroidIndex]);

        // Wait for all local threads
        barrier(CLK_LOCAL_MEM_FENCE);

        // Copy data to global variable
        if(localID < numberOfClusters) {
            atomic_add(&globalSum[localID].r, localSum[localID].r);
            atomic_add(&globalSum[localID].g, localSum[localID].g);
            atomic_add(&globalSum[localID].b, localSum[localID].b);
            atomic_add(&globalSum[localID].a, localSum[localID].a);
            atomic_add(&globalN[localID], localN[localID]);
        }
    }
}

__kernel void update_centroid_values(__global unsigned char *image,
                                    int width,
                                    int height,
                                    __global struct Point *centroids,
                                    __global struct Point *globalSum,
                                    __global int *globalN,
                                    int numberOfClusters,
                                    ulong randoms) 
{
    int globalID = get_global_id(0);
    
    if(globalID < numberOfClusters) {
        // If there is no elements in the cluster, we append one random sample
        if(globalN[globalID] == 0) {
            int min = 0;
            int max = width * height;
            ulong seed = randoms + globalID;
            int r = random_integer(seed, min, max) * 4;

            globalSum[globalID].r = image[r + 2];
            globalSum[globalID].g = image[r + 1];
            globalSum[globalID].b = image[r + 0];
            globalSum[globalID].a = image[r + 3];

            globalN[globalID]++;
        }

        // Set centroid RGBA values by dividing it's sum by corresponding number of elements inside cluster
        int sampleCount       = globalN[globalID];
        centroids[globalID].r = globalSum[globalID].r / sampleCount;
        centroids[globalID].g = globalSum[globalID].g / sampleCount;
        centroids[globalID].b = globalSum[globalID].b / sampleCount;
        centroids[globalID].a = globalSum[globalID].a / sampleCount;
    }
}

// Rebuild image using centroid data
__kernel void rebuild_image(__global unsigned char *image,
                            int width,
                            int height,
                            __global struct Point *centroids,
                            __global int *c) 
{
    int globalID = get_global_id(0);

    if(globalID < width * height) {

        struct Point point = centroids[c[globalID]];
        
        int baseIndex = globalID * 4;
        image[baseIndex + 2] = point.r;
        image[baseIndex + 1] = point.g;
        image[baseIndex + 0] = point.b;
        image[baseIndex + 3] = point.a;
    }
}


/**
 *   @brief Returns Euclidean distance between two points 
 *
 *   @param pointA one struct Point
 *   @param pointB the other struct Point
 *
 *   @return integer that represents Euclidean distance between two 4D points 
 */
int euclidean_distance(struct Point pointA, struct Point pointB) {
    return (pointA.r - pointB.r) * (pointA.r - pointB.r) + 
           (pointA.g - pointB.g) * (pointA.g - pointB.g) + 
           (pointA.b - pointB.b) * (pointA.b - pointB.b) + 
           (pointA.a - pointB.a) * (pointA.a - pointB.a);
}


/**
 *   @brief Returns the random integer in given range
 *
 *   @param seed unsigned long seed
 *   @param min one integer value
 *   @param max the other integer value
 *
 *   @return Random integer that is greather or equal to min and smaller than max value 
 */
int random_integer(ulong seed, int min, int max) {
    seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
    return ((seed >> 16) % (max - min)) + min;
}