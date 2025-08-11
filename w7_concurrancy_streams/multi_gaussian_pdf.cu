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
const int NUM_GPUS = 1;

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
    //allocate memory
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    float *d_input[NUM_GPUS];
    float *d_output[NUM_GPUS];
    
    //Initialize gpu memory
    for(int gpu = 0; gpu < NUM_GPUS; gpu++) {
        cudaSetDevice(gpu);
        cudaMalloc((void**)&d_input[gpu], N * sizeof(float));
        cudaMalloc((void**)&d_output[gpu], N * sizeof(float));
    }
    cudaCheckError("cudaMalloc failed");

    //Generate data in host (CPU)
    for(int gpu = 0; gpu < NUM_GPUS; gpu++) {
        cudaSetDevice(gpu);
        for(int i = 0; i < N; i++) {
            h_input[i] = rand() / (float)RAND_MAX;
        }
        cudaSetDevice(gpu);
        cudaMemcpy(d_input[gpu], h_input, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaCheckError("cudaMemcpy failed");
    }
    
    unsigned long long et1 = dtime_usec(0);

    //Kernel call
    for(int gpu = 0; gpu < NUM_GPUS; gpu++) {
        cudaSetDevice(gpu);
        gaussianPdf<<<(N+255)/256, 256>>>(d_input[gpu], d_output[gpu], 0.0f, 1.0f, N);
        cudaCheckError("cuda kernel launch failed");
    }
    cudaDeviceSynchronize();
    cudaCheckError("cudaDeviceSynchronize failed");
    unsigned long long et2 = dtime_usec(et1);
    
    printf("Time taken: %f usec\n", et2/(float)USECPSEC);

    //Copy data back to host (CPU)
    for(int gpu = 0; gpu < NUM_GPUS; gpu++) {
        cudaSetDevice(gpu);
        cudaMemcpy(h_output, d_output[gpu], N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaCheckError("cudaMemcpy failed");
    }

    //Validate results
    for(int gpu = 0; gpu < NUM_GPUS; gpu++) {
        cudaSetDevice(gpu);
        for(int i = 0; i < N; i++) {
            if (!isEqual(h_output[i], gpdf(h_input[i], 0.0f, 1.0f))) {
                printf("results mismatch at %d, was: %f, should be: %f\n", i, h_output[i], gpdf(h_input[i], 0.0f, 1.0f));
                return 1;
            }
        }
    }
    printf("\033[32mresults passed!\n\033[0m");

    
    //free memory
    free(h_input);
    free(h_output);
    for(int gpu = 0; gpu < NUM_GPUS; gpu++) {
        cudaSetDevice(gpu);
        cudaFree(d_input[gpu]);
        cudaFree(d_output[gpu]);
    }
    cudaCheckError("cudaFree failed");
    return 0;
}