#include <iostream>
#include <cstdlib>

//Macro definition
#define cudaCheckError(msg) {\
    cudaError_t err = cudaGetLastError();\
    if (err != cudaSuccess) {\
        printf("\033[31m%s: %s\n\033[0m", msg, cudaGetErrorString(err));\
        exit(1);\
    }\
}

//Constant definition
const size_t N = 32ULL*1024ULL*1024ULL;
const size_t BLOCK_SIZE = 256;

//Templete 
template <typename T>
//allocate bytes
void allocateBytes(T* &ptr, size_t bytes) {
    cudaMallocManaged((void**)&ptr, bytes);
    cudaCheckError("cudaMallocManaged failed");
}

//kernel function
__global__ void arrayIncrement(int* arr,  size_t n) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < n) {
        arr[idx] += 1;
        idx += blockDim.x * gridDim.x;
    }
}


//main function
int main() {
    //allocate memory
    int* h_arr;
    allocateBytes(h_arr, N * sizeof(int));
    //initialize array
    for (size_t i = 0; i < N; i++) {
        h_arr[i] = i;
    }
    //launch kernel
    arrayIncrement<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(h_arr, N);
    cudaCheckError("kernel launch failed");
    //synchronize
    cudaDeviceSynchronize();
    //validate results
    for (size_t i = 0; i < N; i++) {
        if (h_arr[i] != i + 1) {
            printf("\033[31mresults mismatch at %lu, was: %d, should be: %d\n\033[0m", i, h_arr[i], (int)i + 1); 
            return 1;
        }
    }
    printf("\033[32mValidation passed!\n\033[0m");
    //free memory
    cudaFree(h_arr);
    return 0;
}
