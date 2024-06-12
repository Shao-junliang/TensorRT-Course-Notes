#include "cuda-runtime-api.h"
#include "vector_types.h"

__global__ void demo_kernel() {
    // 打印gird里面x方向上第0列，所有block的x方向上第0列的索引；
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Run kernel. blockIdx = %d,%d,%d  threadIdx = %d,%d,%d\n",
               blockIdx.x, blockIdx.y, blockIdx.z,
               threadIdx.x, threadIdx.y, threadIdx.z);
    }
}

void print_layout(int *girds, int *blocks) {
    dim3 grid_dims(girds[0], girds[1], girds[2]);
    dim3 block_dims(blocks[0], blocks[1], blocks[2]);
    demo_kernel<<<grid_dims, block_dims, 0, nullptr>>>();
}