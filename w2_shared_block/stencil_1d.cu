#include<stdio.h>   
#include<algorithm>

using namespace std;

//Macro definition
#define N 4096
#define BLOCK_SIZE 16
#define RADIUS 2

//Error checking macro
#define cudaCheckError(msg) {\
    cudaError_t err = cudaGetLastError();\
    if (err != cudaSuccess) {\
        printf("%s: %s\n", msg, cudaGetErrorString(err));\
        exit(1);\
    }\
}   

//Kernel function
__global__ void stencil1D(int *input, int *output) {
    __shared__ int sharedMem[BLOCK_SIZE + 2 * RADIUS];
    int gIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int lIdx = threadIdx.x + RADIUS;    
    //read input elements into shared memory
    sharedMem[lIdx] = input[gIdx];
    if (gIdx < N) {
        sharedMem[lIdx- RADIUS] = input[gIdx- RADIUS];
        sharedMem[lIdx+ RADIUS] = input[gIdx+ RADIUS];
    }
    //synchronize threads in the block
    __syncthreads();

    //Apply stencil
    int result = 0;
    for (int i = -RADIUS; i <= RADIUS; i++) {
        result += sharedMem[lIdx + i];
    }
    output[gIdx] = result;
}

void fill_ints(int *x, int n) {
    fill_n(x, n, 1);
}

int main() {
    //pointers to host memory
    int *h_input, *h_output;
    //pointers to device (GPU) memory
    int *d_input, *d_output;
    //size
    int size = (N + 2*RADIUS) * sizeof(int);
    //allocate memory on host
    h_input = (int *)malloc(size);
    h_output = (int *)malloc(size);
    //allocate memory on device (GPU)
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    //check for errors
    cudaCheckError("cudaMalloc failed");
    //Generate data in host (CPU)
    fill_ints(h_input, N+2*RADIUS);
    fill_ints(h_output, N+2*RADIUS);
    //Copy data to device (GPU)
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, size, cudaMemcpyHostToDevice);
    cudaCheckError("cudaMemcpy failed");
    //Calculate grid size to make sure sufficient threads to accomodate all the elements
    //launch kernel
    dim3 block(BLOCK_SIZE);
    dim3 grid(N / block.x);
    stencil1D<<<grid, block>>>(d_input + RADIUS, d_output + RADIUS);
    cudaCheckError("kernel launch failed");
    //Copy data back to host (CPU)
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    cudaCheckError("cudaMemcpy failed");
    //print result
    for (int i = 0; i < 10; i++) {
        printf("Element %i: %d\n", i, h_output[i]);
    }
    //Stencil error checking
    for (int i = 0; i < N + 2*RADIUS; i++) {
        if (i<RADIUS || i>=N+RADIUS){
          if (h_output[i] != 1)
            printf("Mismatch at index %d, was: %d, should be: %d\n", i, h_output[i], 1);
        } else {
          if (h_output[i] != 1 + 2*RADIUS)
            printf("Mismatch at index %d, was: %d, should be: %d\n", i, h_output[i], 1 + 2*RADIUS);
        }
    }
    //free memory on host
    free(h_input);
    free(h_output);
    //free memory on device (GPU)
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;   
}