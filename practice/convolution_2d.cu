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

__global__ void conv2d_basic(float* input, float* kernel, float* output, 
    int input_height, int input_width, 
    int kernel_size, int output_height, int output_width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_height && col < output_width) {
        float sum = 0.0f;
        int kernel_radius = kernel_size / 2;

        for (int kr = 0; kr < kernel_size; kr++) {
            for (int kc = 0; kc < kernel_size; kc++) {
                int input_row = row + kr - kernel_radius;
                int input_col = col + kc - kernel_radius;

                if (input_row >= 0 && input_row < input_height &&
                    input_col >= 0 && input_col < input_width) {
                    sum += input[input_row * input_width + input_col] * 
                        kernel[kr * kernel_size + kc];
            }
        }
    }
    output[row * output_width + col] = sum;
    }
}