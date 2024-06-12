#include "cuda-runtime-api.h"

__global__ void vector_add_kernel(const float *a, const float *b, float *c, int ndata) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= ndata) { return; }
    c[idx] = a[idx] + b[idx];
}

void vector_add(const float *a, const float *b, float *c, int ndata) {
    const int nthreads = 512;
    // 最大线程为 nthreads, 如果
    int block_size = ndata < nthreads ? ndata : nthreads;

    // 这块感觉 grid_size 写一就好了，因为不论ndata与block_size为多少，计算结果都为1；
    int grid_size = (ndata + block_size - 1) / block_size;
    printf("block_size = %d, grid_size = %d\n", block_size, grid_size);

    vector_add_kernel<<<grid_size, block_size, 0, nullptr>>>(a, b, c, ndata);

    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess) {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("kernel error %s:%d  test_print_kernel failed. \n  code = %s, message = %s\n", __FILE__, __LINE__, err_name, err_message);
    }
}