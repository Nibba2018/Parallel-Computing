#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>

__global__ void vectorAdd(float*, float*, float*, int);

//-------------------------------------------------------------

__global__
void vectorAdd(float* A, float* B, float *C, int n)
{
    //CUDA kernel defination

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i<n)
        C[i] = A[i] + B[i];
}

//---------------------------------------------------------------

void vecAdd(float* h_A, float* h_B, float* h_C, long n)
{
    // Host Program

    int size = n * sizeof(float);
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;

    // Error checking object

    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void**)&d_A, size);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&d_B, size);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&d_C, size);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copying input data from host to CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (n+threadsPerBlock-1)/threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", threadsPerBlock, blocksPerGrid);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    err = cudaGetLastError();

    // device function (CUDA kernel) called from host does not have return type.
    // CUDA runtime functions executing in host side can have return types.

    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to Launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy output data from CUDA device to Host\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    // Verifying result:

    for(int i = 0; i < n; i++)
    {
        if(fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");
}

int main()
{
    long n; 
    float *h_A, *h_B, *h_C;

    printf("Enter the size of arrays:");
    scanf("%ld", &n);

	// Initialising Arrays with garbage values for testing purposes
	h_A = (float*)malloc(n*sizeof(float));
	h_B = (float*)malloc(n*sizeof(float));
	h_C = (float*)malloc(n*sizeof(float));    	

	vecAdd(h_A, h_B, h_C, n);

    return 0;
}
