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

//Constant definition
const int DSIZE = 8192;
const int BLOCK_SIZE = 32;
const float A_VAL = 3.0;
const float B_VAL = 2.0;

//Matrix multiply kernel
__global__ void matmulOptimized(float *A, float *B, float *C, int ds) {
    //declare shared memory
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    //x, y indices
    // int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int tx  = threadIdx.x;  
    int ty  = threadIdx.y;  
    int col = blockIdx.x*blockDim.x + tx;  
    int row = blockIdx.y*blockDim.y + ty; 
    
    if(row < ds && col < ds){
        float sum = 0;
        for(int i = 0; i < ds/BLOCK_SIZE; i++){
            //Load data into shared memory
            // Load A-tile: same global row, but local tile-column = tx
            As[ty][tx] = A[row * ds + (i * BLOCK_SIZE + tx)];
            // Load B-tile: same global column, but local tile-row = ty
            Bs[ty][tx] = B[(i * BLOCK_SIZE + ty) * ds + col];
            //synchronize threads in the block
            __syncthreads();
            //Perform matrix multiplication
            for(int k = 0; k < BLOCK_SIZE; k++){    
                sum += As[ty][k] * Bs[k][tx];
            }
            __syncthreads();
        }
        C[row * ds + col] = sum;
    }
}

int main() {
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

    h_A = (float *)malloc(DSIZE * DSIZE * sizeof(float));
    h_B = (float *)malloc(DSIZE * DSIZE * sizeof(float));
    h_C = (float *)malloc(DSIZE * DSIZE * sizeof(float));

    //Generate data in host (CPU)
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        h_A[i] = A_VAL;//rand() / (float)RAND_MAX;
        h_B[i] = B_VAL;//rand() / (float)RAND_MAX;
        h_C[i] = 0;
    }
    
    // Initialization GPU timing
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    //allocate memory on device (GPU)
    cudaMalloc((void **)&d_A, DSIZE * DSIZE * sizeof(float));
    cudaMalloc((void **)&d_B, DSIZE * DSIZE * sizeof(float));
    cudaMalloc((void **)&d_C, DSIZE * DSIZE * sizeof(float));
    //check for errors
    cudaCheckError("cudaMalloc failed");

  
    //Copy data to device (GPU)
    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError("cudaMemcpy failed");
    //Calculate grid size to make sure sufficient threads to accomodate all the elements
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);//block size
    dim3 grid(DSIZE + block.x - 1 / block.x, DSIZE / block.y);//grid size standard way
    //launch kernel
    matmulOptimized<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckError("kernel launch failed");
    //Copy data back to host (CPU)
    cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
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
    for (int i = 0, count=0; i < DSIZE*DSIZE; i++) {
        if (h_C[i] != A_VAL*B_VAL*DSIZE) {
            printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_VAL*B_VAL*DSIZE); 
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
