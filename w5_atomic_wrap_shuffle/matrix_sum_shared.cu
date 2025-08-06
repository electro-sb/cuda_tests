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
const size_t DSIZE = 16384;
const size_t BLOCK_SIZE = 256;
const size_t PRINT_RESULTS = 2;

//matrix row sum kernel
__global__ void matrixRowSum(const float *input, float *output, size_t n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float sharedMem[BLOCK_SIZE];
    sharedMem[threadIdx.x] = 0.0f;
    if (idx < n) {
        float sum = 0.0f;
        //loop over columns
        for (size_t col = 0; col < n; col++) {
            sum += input[idx * n + col];
        }
        sharedMem[threadIdx.x] = sum;
    }
    __syncthreads();
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        __syncthreads();
        if (threadIdx.x < s)  // parallel sweep reduction
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + s];
        }
     if (threadIdx.x == 0) output[blockIdx.x] = sharedMem[0];
}

//matrix column sum kernel
__global__ void matrixColumnSum(const float *input, float *output, size_t n) {
    int idx = threadIdx.x+blockDim.x*blockIdx.x; // create typical 1D thread index from built-in variables
    if (idx < n){
      float sum = 0.0f;
      for (size_t i = 0; i < n; i++)
        sum += input[idx+n*i];         // write a for loop that will cause the thread to iterate down a column, keeeping a running sum, and write the result to sums
      output[idx] = sum;
  
    }
}

//validate results
bool validateResults(float *data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (data[i] != (float)n){
            printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i], (float)n); 
            return false;
        }
    }
    return true;
}
    
int main() {
    float *h_A, *h_sum, *d_A, *d_sum;
    h_A = (float *)malloc(DSIZE * DSIZE * sizeof(float));
    h_sum = (float *)malloc(DSIZE * sizeof(float));

    //Initialize the matrix with 1
    for (size_t i = 0; i < DSIZE * DSIZE; i++) {
        h_A[i] = 1.0f;
    }
    
    //Device memory allocation
    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc(&d_sum, DSIZE * sizeof(float));
    cudaCheckError("cudaMalloc failed");
    //Copy data to device (GPU)
    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError("cudaMemcpy failed");

    //Timing Row sum kernel
    clock_t t0, t1;
    double t1sum, t2sum;
    t0 = clock();
    //launch kernel
    matrixRowSum<<<DSIZE, BLOCK_SIZE>>>(d_A, d_sum, DSIZE);
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf ("Row sum took %f seconds\n", t1sum);
    cudaCheckError("kernel launch failed");

    //Copy data back to host (CPU)
    cudaMemcpy(h_sum, d_sum, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError("cudaMemcpy failed");

    //print result
    for (size_t i = 0; i < PRINT_RESULTS; i++) {
        printf("\033[34mElement %lu: row sum %f\n\033[0m", i, h_sum[i]);
    }

    //Timing Column sum kernel
    t0 = clock();
    //launch kernel
    matrixColumnSum<<<(DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sum, DSIZE);
    t1 = clock();
    t2sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf ("Column sum took %f seconds\n", t2sum); 
    cudaCheckError("kernel launch failed");

    //Copy data back to host (CPU)
    cudaMemcpy(h_sum, d_sum, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError("cudaMemcpy failed");

    //print result
    for (size_t i = 0; i < PRINT_RESULTS; i++) {
        printf("\033[34mElement %lu: column sum %f\n\033[0m", i, h_sum[i]);
    }

    //Verify results
    if (!validateResults(h_sum, DSIZE)) {
        printf("\033[31mValidation failed!\n\033[0m");
        return -1;
    }else{
        printf("\033[32mValidation passed!\n\033[0m");
        printf("\033[32mColumn sum is %f %% faster than row sum\n\033[0m", ((t1sum -t2sum)/t2sum)*100.0);
    }

    //free memory on host
    free(h_A);
    free(h_sum);
    //free memory on device (GPU)
    cudaFree(d_A);
    cudaFree(d_sum);
    return 0;   
}
