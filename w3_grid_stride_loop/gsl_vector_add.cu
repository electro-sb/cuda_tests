#include <stdio.h>

//Macro definition
#define cudaCheckError(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

//constant definition
const int DSIZE = 1024;
const int BLOCK_SIZE = 16;  

__global__ void vectorAddition(const float *input1, const float *input2, float *output, int n) {
   //grid stride loop implementation
   int initial = threadIdx.x + blockDim.x*blockIdx.x;
   int stride = blockDim.x * gridDim.x;
   for (int i = initial; i < n; i += stride) {
       output[i] = input1[i] + input2[i];
   }
}

float randomFloat() {
    return (float)rand()/(float)1000;
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
    for (int i = 0; i < 10; i++) {
        printf("Element %d:h_A= %f, h_B= %f, h_C= %f\n", i, h_A[i], h_B[i], h_C[i]);
    }
    // Verify results
    cudaCheckError("kernel execution failure or cudaMemcpy H2D failure");
    for (int i = 0, count=0; i < DSIZE; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], h_A[i] + h_B[i]); 
            count++;
            if(count > 10)
                break;
        }
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
