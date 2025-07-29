#include<stdio.h>   

//Error checking macro
//For checking errors in any CUDA operation
#define cudaCheckError(msg) {\
    cudaError_t err = cudaGetLastError();\
    if (err != cudaSuccess) {\
        printf("%s: %s\n", msg, cudaGetErrorString(err));\
        exit(1);\
    }\
}

//Constant definition
const int DSIZE = 4096;
const int BLOCK_SIZE = 16;

__global__ void matmulSimple(float *A, float *B, float *C) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx < DSIZE && idy < DSIZE) {
        float sum = 0;
        for (int i = 0; i < DSIZE; i++) {
            sum += A[idx * DSIZE + i] * B[i * DSIZE + idy];
        }
        C[idx * DSIZE + idy] = sum;
    }
}

int main() {
    //pointers to host memory
    float *h_A, *h_B, *h_C;
    //pointers to device (GPU) memory
    float *d_A, *d_B, *d_C;
    //allocate memory on host
    h_A = (float *)malloc(DSIZE * DSIZE * sizeof(float));
    h_B = (float *)malloc(DSIZE * DSIZE * sizeof(float));
    h_C = (float *)malloc(DSIZE * DSIZE * sizeof(float));
    //allocate memory on device (GPU)
    cudaMalloc((void **)&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc((void **)&d_B, DSIZE * DSIZE * sizeof(float));
    cudaMalloc((void **)&d_C, DSIZE * DSIZE * sizeof(float));
    //check for errors
    cudaCheckError("cudaMalloc failed");
    //Generate data in host (CPU)
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = 0;
    }
    //Copy data to device (GPU)
    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError("cudaMemcpy failed");
    //Calculate grid size to make sure sufficient threads to accomodate all the elements
    //launch kernel
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);//block size
    dim3 grid(DSIZE / block.x, DSIZE / block.y);//grid size standard way
    
    matmulSimple<<<grid, block>>>(d_A, d_B, d_C);
    cudaCheckError("kernel launch failed");
    //Copy data back to host (CPU)
    cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError("cudaMemcpy failed");
    //print result
    for (int i = 0; i < 10; i++) {
        printf("Element %i: %f\n", i, h_C[i]);
    }
    //free memory on host
    free(h_A);
    free(h_B);
    free(h_C);
    //free memory on device (GPU)
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;   
}
     
