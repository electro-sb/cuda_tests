
#include <stdio.h>

__global__ void hello() {
    printf("Hello, from Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}

//main function on host
int main() {
    //launch cuda kernel
    //grid of 1 block, each block has 1 thread
    hello<<<1, 1>>>();
    //synchronize   
    cudaDeviceSynchronize();
    return 0;
}