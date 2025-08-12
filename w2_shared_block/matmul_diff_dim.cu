#include<stdio.h>
//Macro definition
// error checking macro
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

#ifndef TILE
#define TILE 32 // set once; must match launch or use blockDim.{x,y} below
#endif
    

//Constant definition
//const int DSIZE = 8192;
//const int BLOCK_SIZE = 32;
const float A_VAL = 3.0;
const float B_VAL = 2.0;

//Matrix multiply kernel A Rank MxK, B Rank KxN, C Rank MxN
__global__ void matmulOptimized(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K) {
    // Use compile-time TILE for static shared memory (fast), but ensure launch matches.
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    // Number of K-tiles (ceil division)
    int tilesK = (K + TILE - 1) / TILE;

    float sum = 0.0f;

    for (int t = 0; t < tilesK; ++t) {
        int kA = t * TILE + tx; // column into A
        int kB = t * TILE + ty; // row into B

        // Zero-pad guarded loads (all threads participate)
        As[ty][tx] = (row < M && kA < K) ? A[row * K + kA] : 0.0f;
        Bs[ty][tx] = (kB < K && col < N) ? B[kB * N + col] : 0.0f;

        __syncthreads();

        // Accumulate this tile
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    int M = 2048, N = 1024, K = 512;   
    //pointers to host memory
    float *h_A, *h_B, *h_C;
    //pointers to device (GPU) memory
    float *d_A, *d_B, *d_C;
    //allocate memory on host
    clock_t t0, t1, t2;
    double t1sum, t2sum;
    t0 = clock();
    //status variable
    int status = 0;

    h_A = (float *)malloc(M * K * sizeof(float));
    h_B = (float *)malloc(K * N * sizeof(float));
    h_C = (float *)malloc(M * N * sizeof(float));

    //Generate data in host (CPU)
    for (int i = 0; i < M*K; ++i) h_A[i] = A_VAL;
    for (int i = 0; i < K*N; ++i) h_B[i] = B_VAL;        // FIX
    for (int i = 0; i < M*N; ++i) h_C[i] = 0.0f; 
    
    // Initialization GPU timing
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    //allocate memory on device (GPU)
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));
    //check for errors
    cudaCheckError("cudaMalloc failed");

  
    //Copy data to device (GPU)
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError("cudaMemcpy failed");
    //Calculate grid size to make sure sufficient threads to accomodate all the elements
    dim3 block(TILE, TILE);//block size TILE
    dim3 grid(  (N + TILE - 1) / TILE, (M + TILE - 1) / TILE);//grid size standard way
    //launch kernel
    matmulOptimized<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaCheckError("kernel launch failed");
    //Copy data back to host (CPU)
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError("cudaMemcpy failed");

    // Compute GPU timing
    t2 = clock();
    t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
    printf ("Done. Compute took %f seconds\n", t2sum);

    //print result
    for (int i = 0; i < 10; i++) {
        printf("Element %d:h_A= %f, h_B= %f, h_C= %f\n", i, h_A[i], h_B[i], h_C[i]);
    }
    // Verify results
    cudaCheckError("kernel execution failure or cudaMemcpy H2D failure");
    for (int i = 0, count=0; i < M*N; i++) {
        if (h_C[i] != A_VAL*B_VAL*K) {
            printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_VAL*B_VAL*K); 
            status = -1;
            if(count++ > 10)
                break;
        }
    }
    if(status == 0){
        printf("Success!\n"); 
    }
    else{
        printf("Failure!\n");
    }   

    //free memory on host
    free(h_A);
    free(h_B);
    free(h_C);
    //free memory on device (GPU)
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return status;  
}
  