#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <sys/time.h>

//Macro definition
#define cudaCheckError(msg) do{\
    cudaError_t err = cudaGetLastError();\
    if (err != cudaSuccess) {\
        printf("\033[31m%s: %s\n\033[0m", msg, cudaGetErrorString(err));\
        exit(1);\
    }\
} while (0)

// host-based timing
#define USECPSEC 1000000ULL
// host-based timing function
unsigned long long dtime_usec(unsigned long long start) {
    timeval tv;
    gettimeofday(&tv, 0);
    return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

//constant definition
const int CHUNKS = 64;
const size_t N = 1024ULL*1024ULL*CHUNKS;
//const int COUNT = 22;
const int STREAMS = 4;


const float SQRT_2PIF = 2.5066282747946493232942230134974f;
const double SQRT_2PI = 2.5066282747946493232942230134974;

__device__ __host__ float gpdf(float val, float mean, float std) {
    //Gaussian probability density function: 
    // PDF(x) = (1 / (std * sqrt(2 * pi))) * exp(-0.5 * ((x - mean) / std)^2)
    return (1.0f / (std * SQRT_2PIF)) * expf(-0.5f * powf((val - mean) / std, 2.0f));
}

__device__ __host__ double gpdf(double val, double mean, double std) {
    //Gaussian probability density function: 
    // PDF(x) = (1 / (std * sqrt(2 * pi))) * exp(-0.5 * ((x - mean) / std)^2)
    return (1.0 / (std * SQRT_2PI)) * exp(-0.5 * pow((val - mean) / std, 2.0));
}

__host__ bool isEqual(float a, float b, float epsilon= 1e-6) {
    return fabs(a - b) < epsilon;
}

__global__ void gaussianPdf(const float *input, 
                            float *output, 
                            const float mean, 
                            const float std,
                            const int n) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        output[idx] = gpdf(input[idx], mean, std);
    }
}

int main() {
    float *h_input, *h_output;
    
    //allocate memory
    cudaHostAlloc((void**)&h_input, N * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_output, N * sizeof(float), cudaHostAllocDefault);
    float *d_input;
    float *d_output;
    cudaStream_t streams[STREAMS];
    for (int i = 0; i < STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    //Initialize gpu memory
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));
    cudaCheckError("cudaMalloc failed");

    //Generate data in host (CPU)
    for(int i = 0; i < N; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError("cudaMemcpy failed");
    
    unsigned long long et1 = dtime_usec(0);

    //Kernel call
    for(int index=0; index < CHUNKS; index++){
        cudaMemcpyAsync(h_input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost, streams[index % STREAMS]); 
        gaussianPdf<<<(N+255)/256, 256,0,streams[index % STREAMS]>>>(d_input, d_output, 0.0f, 1.0f, N);
        cudaMemcpyAsync(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost, streams[index % STREAMS]); 
    }
    cudaCheckError("cuda kernel launch failed");
    cudaDeviceSynchronize();
    cudaCheckError("cudaDeviceSynchronize failed");
    unsigned long long et2 = dtime_usec(et1);
    
    printf("Time taken: %f usec\n", et2/(float)USECPSEC);

    //Copy data back to host (CPU)
    //cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError("cudaMemcpy failed");

    //Validate results
    for(int i = 0; i < N; i++) {
            if (!isEqual(h_output[i], gpdf(h_input[i], 0.0f, 1.0f))) {
                printf("results mismatch at %d, was: %f, should be: %f\n", i, h_output[i], gpdf(h_input[i], 0.0f, 1.0f));
                return 1;
            }
        }
    printf("\033[32mresults passed!\n\033[0m");

    
    //free memory
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaCheckError("cudaFree failed");
    return 0;   
}   