#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

void print_matrix(float *A,int m,int n)
{
    for(int i =0;i<m;i++)
    {
        for(int j=0;j<n;j++)
            printf("%.1f ",A[i*n+j]);
        printf("\n");
    }

}

__global__ void swapReflect(float *input, float *output, int M, int N)
{
    int j = threadIdx.x;

    for(int i=0; i<M; i++)
    {
        if(j%2 == 0)
        {    
            output[i*N+j] = input[i*N+j+1];
            output[i*N+j+1] = input[i*N+j];
        }
    }
    __syncthreads();

    for(int i = 0; i<j; i++)
    {
        int val = output[j*N + i];
        output[j*N + i] = output[i*N + j];
        output[i*N + j] = val;
    }

}

int main(void)
{
    cudaError_t err = cudaSuccess;

    int t; // No of test Cases
    scanf("%d", &t);

    while(t--)
    {
        int m, n;
        scanf("%d %d", &m, &n);
        size_t size = m*n * sizeof(float);

        //Allocate host input
        float *h_input = (float*)malloc(size);

        //Allocate host output
        float *h_output = (float*)malloc(size);

        // Verify that allocations succeeded
        if (h_input == NULL)
        {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }

        // Initialize the host input 
        
        for (int i = 0; i < n*m; ++i)
        {
            scanf("%f",&h_input[i]);
        }

        float *d_input = NULL, *d_output = NULL;
        
        //Allocate device input
        cudaMalloc((void**)&d_input, size);

        //Allocate device output
        cudaMalloc((void**)&d_output, size);

        //Copy data from host to device
        cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_output, h_output, size, cudaMemcpyHostToDevice);
        
        dim3 grid(1, 1, 1);
        dim3 block(n, 1, 1);

        swapReflect<<<grid, block>>>(d_input, d_output, m, n);
        err = cudaGetLastError();

        if(err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy output of device to host
        cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

        print_matrix(h_output, m, n);

    }

    return 0;
}