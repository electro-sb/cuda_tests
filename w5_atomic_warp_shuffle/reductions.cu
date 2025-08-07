#include <stdio.h>

//error checking macro
#define cudaCheckError(msg) do{\
    cudaError_t err = cudaGetLastError();\
    if (err != cudaSuccess) {\
        printf("%s: %s\n", msg, cudaGetErrorString(err));\
        exit(1);\
    }\
} while (0)

//constants 
const size_t N = 8ULL*1024ULL*1024ULL;//ULL is used to make it unsigned long long
const size_t BLOCK_SIZE = 256;
const size_t GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

//1. naive atomic reduction kernel
__global__ void naiveAtomicReduction(float *input, float *output, size_t N) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        atomicAdd(output, input[idx]);
    }
}

//2. standard reduction kernel
__global__ void standardReduction(float *input, float *output, size_t N) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t tid = threadIdx.x;
    __shared__ float sharedMem[BLOCK_SIZE];
    sharedMem[tid] = 0.0f;
    //load data into shared memory
    while (idx < N) {
        sharedMem[tid] += input[idx];
        idx += blockDim.x * gridDim.x;
    }
    //reduce within the block
    for (size_t s = blockDim.x/2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            sharedMem[tid] += sharedMem[tid + s];
        }
    }
    //store result in global memory
    if (tid == 0) {
        //output[blockIdx.x] = sharedMem[0];//This only adds the block
        atomicAdd(output, sharedMem[0]);
    }
}

//3. standard reduction with atomic
__global__ void atomicReduction(float *input, float *output, size_t N) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float sharedMem[BLOCK_SIZE];
    size_t tid = threadIdx.x;
    sharedMem[tid] = 0.0f;
    //load data into shared memory
    while (idx < N) {
        sharedMem[tid] += input[idx];
        idx += blockDim.x * gridDim.x;
    }
    //reduce within the block
    for (size_t s = blockDim.x/2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            sharedMem[tid] += sharedMem[tid + s];
        }
    }
    //store result in global memory
    if (tid == 0) {
        atomicAdd(output, sharedMem[0]);
    }
}


//4.  reduction with warp shuffle
__global__ void warpShuffleReduction(float *in, float *out, size_t N) {
    // 1) Stride‐loop to sum each thread’s chunk
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    float sum     = 0.0f;
    while (idx < N) {
        sum += in[idx];
        idx += stride;
    }

    // 2) In‐warp reduction to lane 0
    unsigned mask = 0xFFFFFFFFu;
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // 3) Lane 0 of each warp does one atomic add
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(out, sum);
    }
}

int main() {
    //timing variables
    clock_t t1,t2;
    double tdiff;
    
    //initialize random seed
    srand(time(NULL));
    //allocate memory on host
    float *h_input = (float *)malloc(N * sizeof(float));
    float *h_output = (float *)malloc(sizeof(float));
    //initialize input data
    for (size_t i = 0; i < N; i++) {
        h_input[i] = 1.0f;//rand()/(float)RAND_MAX;
    }
    //allocate memory on device
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, sizeof(float));
    cudaMemset(d_output, 0, sizeof(float));

    cudaCheckError("cudaMalloc failed");
    //copy data to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    //1. launch naive atomic reduction kernel
    t1 = clock();
    naiveAtomicReduction<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, N);
    t2 = clock();
    tdiff = ((double)(t2-t1))/CLOCKS_PER_SEC;
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError("cudaMemcpy failed");
    printf("\033[34mNaive atomic reduction output: %0.2f took %f seconds\n\033[0m", *h_output, tdiff);
    
     //2. launch standard reduction kernel
     //copy data to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));
     t1 = clock();
     standardReduction<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, N);
     t2 = clock();
     tdiff = ((double)(t2-t1))/CLOCKS_PER_SEC;
     cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
     cudaCheckError("cudaMemcpy failed");
     printf("\033[34mStandard reduction output: %0.2f took %f seconds\n\033[0m", *h_output, tdiff);

    //3. launch atomic reduction kernel
    //copy data to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));
    t1 = clock();
    atomicReduction<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, N);
    t2 = clock();
    tdiff = ((double)(t2-t1))/CLOCKS_PER_SEC;
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError("cudaMemcpy failed");
    printf("\033[34mAtomic reduction output: %0.2f took %f seconds\n\033[0m", *h_output, tdiff);
    
    //4. launch warpShuffleReduction kernel
    //copy data to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));
    t1 = clock();
    warpShuffleReduction<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, N);
    t2 = clock();
    tdiff = ((double)(t2-t1))/CLOCKS_PER_SEC;
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError("cudaMemcpy failed");
    printf("\033[34mWarp shuffle reduction output: %0.2f took %f seconds\n\033[0m", *h_output, tdiff);
    //copy data back to host
    //print result
    printf("\033[32mResult: %0.2f correct result should be %ld \n\033[0m", *h_output, N);
    //free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);    
    return 0;
}

    
   