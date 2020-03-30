#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>

void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number: %d\n", devProp.major);
    printf("Minor revision number: %d\n", devProp.minor);
    printf("Name: %s\n", devProp.name);
    printf("Total global memory: %lu\n", devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n", devProp.sharedMemPerBlock);
    printf("Total Registers per block: %d\n", devProp.regsPerBlock);
    printf("Warp Size: %d\n", devProp.warpSize);
    printf("Maximum Memory pitch: %lu\n", devProp.memPitch);
    printf("Maximum Threads per block: %d\n", devProp.maxThreadsPerBlock);
    for(int i = 0; i<3; i++)
        printf("Maximum dimension of block %d: %d\n", i, devProp.maxThreadsDim[i]);
    for(int i = 0; i<3; i++)
        printf("Maximum dimension of grid %d: %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate: %d\n", devProp.clockRate);
    printf("Total Constant Memory: %lu\n", devProp.totalConstMem);
    printf("Texture alignment: %lu\n", devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors: %d\n", devProp.multiProcessorCount);
}

int main()
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    for(int i=0; i<devCount; i++)
    {
        cudaDeviceProp devp;
        cudaGetDeviceProperties(&devp, i);
        printDevProp(devp);
    }
    return 0;
}