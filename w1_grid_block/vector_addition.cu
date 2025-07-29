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
const int DSIZE = 1024;
const int BLOCK_SIZE = 256;

__global__ void vectorAddition(const float *input1, const float *input2, float *output, int n) {
    int i = threadIdx.x;
    if (i < n) {
        output[i] = input1[i] + input2[i];
    }
}

float randomFloat() {
    return (float)rand()/(float)1000;//RAND_MAX
}

int main() {
    //pointers to host memory
    float *h_A, *h_B, *h_C;
    //pointers to device (GPU) memory
    float *d_A, *d_B, *d_C;
    //allocate memory on host
    h_A = (float *)malloc(DSIZE * sizeof(float));
    h_B = (float *)malloc(DSIZE * sizeof(float));
    h_C = (float *)malloc(DSIZE * sizeof(float));
    //allocate memory on device (GPU)
    cudaMalloc((void **)&d_A, DSIZE * sizeof(float));
    cudaMalloc((void **)&d_B, DSIZE * sizeof(float));
    cudaMalloc((void **)&d_C, DSIZE * sizeof(float));
    //check for errors
    cudaCheckError("cudaMalloc failed");

    //Generate data in host (CPU)
    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = randomFloat();
        h_B[i] = randomFloat();
        h_C[i] = 0;
    }
    //Copy data to device (GPU)
    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError("cudaMemcpy failed");
    //Calculate grid size to make sure sufficient threads to accomodate all the elements
    //launch kernel
    vectorAddition<<<(DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckError("kernel launch failed");
    //Copy data back to host (CPU)
    cudaMemcpy(h_C, d_C, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError("cudaMemcpy failed");
    //print result
    for (int i = 0; i < 10; i++) {//Print the first 10 results
        printf("Element %i: %f + %f = %f\n", i, h_A[i], h_B[i], h_C[i]);
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