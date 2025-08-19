#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

//Macro definition
#define cudaCheckError(msg) do{\
    cudaError_t err = cudaGetLastError();\
    if (err != cudaSuccess) {\
        printf("\033[31m%s: %s\n\033[0m", msg, cudaGetErrorString(err));\
        exit(1);\
    }\
} while (0)

//Constant definition
const int TILE_DIM = 256;

// Generate random samples instead of evenly spaced
__host__ float* randomSamples(float a, float b, int N) {
    float *arr = (float *)malloc(N * sizeof(float));
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        float r = (float)rand() / RAND_MAX;  // Random number between 0 and 1
        arr[i] = a + (b - a) * r;  // Scale to [a, b]
    }
    return arr;
}

//a function f(x) = x^3 + x^2 + 1, returns an array of size N
__host__ float* f(float *x, int N) {
    float *y = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        y[i] = x[i] * x[i] * x[i] + x[i] * x[i] + 1;
    }
    return y;
}

__global__ void montecarloIntegration(const float* y_samples, float* result, 
                                     float a, float b, int n_samples) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    //shared memory - use dynamic allocation to avoid compile-time constant issues
    extern __shared__ float smem[];
    
    // Initialize shared memory
    smem[tid] = 0.0f;
    
    // Load data into shared memory (handle out-of-bounds)
    if (idx < n_samples) {
        smem[tid] = y_samples[idx];
    }
    __syncthreads();

    // Block-level reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    
    // Only thread 0 of each block writes to global memory
    if (tid == 0) {
        atomicAdd(result, smem[0]);
    }
}

// Alternative version that handles more samples per thread
__global__ void montecarloIntegrationMultiPass(const float* y_samples, float* result, 
                                               float a, float b, int n_samples) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float smem[];
    
    // Each thread processes multiple elements
    float thread_sum = 0.0f;
    for (int i = idx; i < n_samples; i += blockDim.x * gridDim.x) {
        thread_sum += y_samples[i];
    }
    smem[tid] = thread_sum;
    
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    
    // Only thread 0 of each block writes to global memory
    if (tid == 0) {
        atomicAdd(result, smem[0]);
    }
}

int main() {
    //allocate memory on host
    float *h_y, *d_y, *h_result, *d_result;
    float a = 0.0f, b = 2.0f;
    int N = 1024;
    
    // Use random samples for Monte Carlo
    float *x = randomSamples(a, b, N);
    h_y = f(x, N);  // Don't malloc again, just assign
    
    // Allocate device memory
    cudaMalloc((void **)&d_y, N * sizeof(float));
    cudaMalloc((void **)&d_result, sizeof(float));
    
    // CRITICAL: Initialize device result to 0
    cudaMemset(d_result, 0, sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError("cudaMemcpy failed");
    
    // Launch kernel with shared memory
    int blockSize = TILE_DIM;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    montecarloIntegration<<<gridSize, blockSize, blockSize * sizeof(float)>>>
                         (d_y, d_result, a, b, N);
    cudaCheckError("kernel launch failed");
    
    // Copy result back
    h_result = (float *)malloc(sizeof(float));
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError("cudaMemcpy result failed");
    
    // Complete Monte Carlo integration formula: (b-a) * average
    float monte_carlo_result = (b - a) * (*h_result) / N;
    
    printf("Sum of samples: %f\n", *h_result);
    printf("Monte Carlo Integration Result: %f\n", monte_carlo_result);
    
    // For comparison, analytical result of ∫(x³ + x² + 1)dx from 0 to 2
    // = [x⁴/4 + x³/3 + x]₀² = 16/4 + 8/3 + 2 = 4 + 2.667 + 2 = 8.667
    printf("Analytical result: %f\n", 8.6666667f);
    printf("Error: %f\n", fabs(monte_carlo_result - 8.6666667f));
    
    // Cleanup
    free(x);
    free(h_y);
    free(h_result);
    cudaFree(d_y);
    cudaFree(d_result);
    
    return 0;
}

// CPU version for verification
float monte_carlo_cpu(float a, float b, int N) {
    float sum = 0.0f;
    srand(time(NULL));
    
    for (int i = 0; i < N; i++) {
        float x = a + (b - a) * ((float)rand() / RAND_MAX);
        float fx = x * x * x + x * x + 1;
        sum += fx;
    }
    
    return (b - a) * sum / N;
}