#include <cstdlib>
#include <cstdio>

//Macro definition
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "\033[31m%s: %s\n\033[0m", msg, cudaGetErrorString(__err)); \
            exit(1); \
        } \
    } while (0) 

//list element
typedef struct list_element {
    int key;
    struct list_element* next;
} list_element;

//Templete function
template <typename T>
void allocateBytes(T* &ptr, size_t bytes) {
    cudaMallocManaged((void**)&ptr, bytes);
    cudaCheckErrors("cudaMallocManaged failed");
}

//print list element
__host__ __device__ void printListElement(list_element* element, int to_print) {
    list_element* current = element;
    for (int i = 0; i < to_print; i++) {
        current = current->next;
    }
    printf("%d ", current->key);
    printf("\n");
}

//kernel function
__global__ void printListKernel(list_element* element, int to_print) {
    printListElement(element, to_print);
}

//constant definition
const int NUM_ELEMENTS = 5;
const int N = 3;

//main function
int main() {
    //allocate memory
    list_element* h_list;
    allocateBytes(h_list, N * sizeof(list_element));
    //initialize list
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        h_list[i].key = i;
        h_list[i].next = &h_list[i + 1];
    }
    //launch kernel
    printListKernel<<<1, 1>>>(h_list, N);
    cudaCheckErrors("kernel launch failed");
    //synchronize
    cudaDeviceSynchronize();
    //validate results
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        if (h_list[i].key != i) {
            printf("\033[31mresults mismatch at %d, was: %d, should be: %d\n\033[0m", i, h_list[i].key, (int)i); 
            return 1;
        }
    }
    printf("\033[32mValidation passed!\n\033[0m");
    //free memory
    cudaFree(h_list);
    return 0;
}
    