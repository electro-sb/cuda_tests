#include <stdio.h>
#include <iostream>

#ifdef DEBUG
#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);} 
#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);} 
#else
#define CUDA_CALL(F) (F)
#define CUDA_CHECK() 
#endif

#define cudaCheckError(msg) do{\
    cudaError_t err = cudaGetLastError();\
    if (err != cudaSuccess) {\
        printf("\033[31m%s: %s\n\033[0m", msg, cudaGetErrorString(err));\
        exit(1);\
    }\
} while (0)

//Definition of thread block size in x and y direction
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

//Definition of matrix linear size
#define SIZE 4096

//macro to index 2D array in 1 dimention in column major format
#define INDEX(row,col, linear_size) ((col) * (linear_size) + (row))

//define the naive matrix transpose kernel
__global__ void naiveMatrixTranspose(const int m,
                                    const double *const a,
                                    double *const c) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if (row < m && col < m) {
        c[INDEX(col, row, m)] = a[INDEX(row, col, m)];
    }
    return;
}

void hostMatrixTranspose(const int m,
                        const double *const a,
                        double *const c) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < m; col++) {
            c[INDEX(col, row, m)] = a[INDEX(row, col, m)];
        }
    }
    return;
}

int main(int argc, char *argv[]) {
    fprintf(stdout, "\033[32mMatrix size: %d\n\033[0m", SIZE);
    //Allocate memory on host
    double *h_a, *h_c;
    double *d_a, *d_c;
    
    size_t num_bytes = (size_t)SIZE * (size_t)SIZE * sizeof(double);
    h_a = (double *)malloc(num_bytes);
    if(h_a == NULL) {
        fprintf(stderr, "\033[31mFailed to allocate memory on host\n\033[0m");
        return -1;
    }
    h_c = (double *)malloc(num_bytes);
    if(h_c == NULL) {
        fprintf(stderr, "\033[31mFailed to allocate memory on host\n\033[0m");
        return -1;
    }
    
    //Allocating memory on device
    CUDA_CALL(cudaMalloc((void **) &d_a, num_bytes));
    CUDA_CALL(cudaMalloc((void **) &d_c, num_bytes));
    cudaCheckError("cudaMalloc failed");

    //set result matrices to zero
    CUDA_CALL(cudaMemset(d_c, 0, num_bytes));
    cudaCheckError("cudaMemset failed");
    
    //Initilaze the host input matrix with normal values
    for (int row = 0; row < SIZE; row++) {
        for (int col = 0; col < SIZE; col++) {
            h_a[INDEX(row, col, SIZE)] = (double)rand() / (double)RAND_MAX;
        }
    }   
    
    //Copy data to device
    CUDA_CALL(cudaMemcpy(d_a, h_a, num_bytes, cudaMemcpyHostToDevice));
    cudaCheckError("cudaMemcpy failed");
    
    //start and stop timer events
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    cudaCheckError("cudaEventCreate failed");

    CUDA_CALL(cudaEventRecord(start, 0));
    //host transpose
    hostMatrixTranspose(SIZE, h_a, h_c);
    CUDA_CALL(cudaEventRecord(stop, 0));
    cudaCheckError("cudaEventRecord failed");

    CUDA_CALL(cudaEventSynchronize(stop));
    float elapsed_time;
    CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));
    cudaCheckError("cudaEventElapsedTime failed");
    fprintf(stdout, "\033[32mHost transpose took %f ms\n\033[0m", elapsed_time);
    
    //Launch kernel
    dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 grid((SIZE + block.x - 1) / block.x, (SIZE + block.y - 1) / block.y);
    naiveMatrixTranspose<<<grid, block>>>(SIZE, d_a, d_c);
    //CUDA_CHECK();
    cudaCheckError("kernel launch failed");
    
    //Copy data back to host
    CUDA_CALL(cudaMemcpy(h_c, d_c, num_bytes, cudaMemcpyDeviceToHost));
    cudaCheckError("cudaMemcpy failed");
    
    //Validate results
    for (int row = 0; row < SIZE; row++) {
        for (int col = 0; col < SIZE; col++) {
            if (h_c[INDEX(row, col, SIZE)] != h_a[INDEX(col, row, SIZE)]) {
                fprintf(stderr, "\033[31mresults mismatch at %d, was: %f, should be: %f\n\033[0m", INDEX(row, col, SIZE), h_c[INDEX(row, col, SIZE)], h_a[INDEX(col, row, SIZE)]);
                return -1;
            }
        }
    }
    
    //Free memory
    free(h_a);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}