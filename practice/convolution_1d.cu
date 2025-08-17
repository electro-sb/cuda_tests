#include <stdio.h>

//Macro definition
#define cudaCheckError(msg) do{\
    cudaError_t err = cudaGetLastError();\
    if (err != cudaSuccess) {\
        printf("\033[31m%s: %s\n\033[0m", msg, cudaGetErrorString(err));\
        exit(1);\
    }\
} while (0)

//Constant definition
const int THREADS_PER_BLOCK = 256;

__global__ void convolution1D(const float *input, const float *kernel, float *output, int input_size, int kernel_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int output_size = input_size - kernel_size + 1;
    if(idx < output_size) {
        float sum = 0.0f;
        
        // For valid convolution: output[idx] corresponds to input starting at idx
        for(int k = 0; k < kernel_size; ++k) {
            int input_idx = idx + k;  // Start at idx, go forward
            if(input_idx < input_size) {
                sum += input[input_idx] * kernel[k];
            }
        }
        output[idx] = sum;
    }
}


__global__ void convolution1D_explicit(const float* input, const float* kernel, float* output,
    int input_size, int kernel_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;

    if(idx < output_size) {
        float sum = 0.0f;

        for(int k = 0; k < kernel_size; ++k) {
            sum += input[idx + k] * kernel[k];
        }
        output[idx] = sum;
    }
}


int main() {
    const int input_size = 512;
    const int kernel_size = 3;
    int output_size = input_size - kernel_size + 1;
    //host memory
    float *h_input = (float *)malloc(input_size * sizeof(float));
    float *h_kernel = (float *)malloc(kernel_size * sizeof(float));
    float *h_output = (float *)malloc(output_size * sizeof(float));

    for (int i = 0; i < input_size; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < kernel_size; i++) {
        h_kernel[i] = i + 0.5;
    }

    //device memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc((void **)&d_input, input_size * sizeof(float));
    cudaMalloc((void **)&d_kernel, kernel_size * sizeof(float));
    cudaMalloc((void **)&d_output, output_size * sizeof(float));
    cudaCheckError("cudaMalloc failed");
    
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError("cudaMemcpy failed");
    
    int blocks = (input_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    convolution1D_explicit<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_kernel, d_output, input_size, kernel_size);
    cudaCheckError("convolution1D");
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError("cudaMemcpy");
    for (int i = 0; i < output_size; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");
    free(h_input);
    free(h_kernel);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    return 0;
}
