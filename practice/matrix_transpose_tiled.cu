#include <stdio.h>

//Macro definition
#define cudaCheckError(msg) do{\
    cudaError_t err = cudaGetLastError();\
    if (err != cudaSuccess) {\
        printf("\033[31m%s: %s\n\033[0m", msg, cudaGetErrorString(err));\
        exit(1);\
    }\
} while (0) 

#define TILE_DIM 16
#define BLOCK_ROWS 16
#define ROW_SIZE 4096
#define COL_SIZE 1024

__global__ void transposeTiled(float *odata, const float *idata, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; 
    // +1 padding avoids shared mem bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile from global to shared (row-major, coalesced)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = idata[y * width + x];
    }

    __syncthreads();

    // Transpose block offset: swap x/y roles
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write tile from shared to global (now column-major in original, coalesced in write)
    if (x < height && y < width) {
        odata[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
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
    cudaMalloc((void **)&d_input, ROW_SIZE * COL_SIZE * sizeof(float));
    cudaMalloc((void **)&d_output, ROW_SIZE * COL_SIZE * sizeof(float));
    cudaCheckError("cudaMalloc failed");
    //copy data from host to device
    cudaMemcpy(d_input, h_input, ROW_SIZE * COL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    //launch kernel
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((COL_SIZE + TILE_DIM - 1) / TILE_DIM, 
              (ROW_SIZE + TILE_DIM - 1) / TILE_DIM);

    printf("block=(%d,%d) grid=(%d,%d)\n", block.x, block.y, grid.x, grid.y);
    transposeTiled<<<grid, block>>>(d_output, d_input, COL_SIZE, ROW_SIZE);
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