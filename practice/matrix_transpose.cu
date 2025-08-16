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
const int ROW_SIZE = 4096;
const int COL_SIZE = 1024;
//const int BLOCK_SIZE = 256;
const int BLOCK_X = 16;
const int BLOCK_Y = 16;

__global__ void matrixTranspose(float *input, float *output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
    //Cuda stores matrices in Row major format
}

int main() {
    //allocate memory on host
    float *h_input = (float *)malloc(ROW_SIZE * COL_SIZE * sizeof(float));
    float *h_output = (float *)malloc(ROW_SIZE * COL_SIZE * sizeof(float));
    //generate data in host (CPU)
    for (int i = 0; i < ROW_SIZE * COL_SIZE; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }
    
    //allocate memory on device (GPU)
    float *d_input, *d_output;
    cudaMalloc(&d_input, ROW_SIZE * COL_SIZE * sizeof(float));
    cudaMalloc(&d_output, ROW_SIZE * COL_SIZE * sizeof(float));
    cudaCheckError("cudaMalloc failed");
    //copy data from host to device
    cudaMemcpy(d_input, h_input, ROW_SIZE * COL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    //launch kernel
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((COL_SIZE + block.x - 1) / block.x, 
              (ROW_SIZE + block.y - 1) / block.y);

    printf("block=(%d,%d) grid=(%d,%d)\n", block.x, block.y, grid.x, grid.y);
    matrixTranspose<<<grid, block>>>(d_input, d_output, ROW_SIZE, COL_SIZE);
    cudaCheckError("kernel launch failed: ");
    //copy data back to host
    cudaMemcpy(h_output, d_output, ROW_SIZE * COL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError("cudaMemcpy failed ");
    //validate results
    for (int row = 0; row < ROW_SIZE; row++) {
        for (int col = 0; col < COL_SIZE; col++) {
            float a_element = h_input[row*COL_SIZE + col];
            float b_element = h_output[col*ROW_SIZE + row];
            if (a_element != b_element) {
                printf("Mismatch at index [%d, %d], was: %f, should be: %f\n", row, col, a_element, b_element);
                return -1;
            }
        }
    }
    printf("Success!\n");
    //free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}