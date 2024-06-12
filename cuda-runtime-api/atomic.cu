#include "cuda-runtime-api.h"

__global__ void launch_keep_item_kernel(float *input_array, int input_size, float threshold, float *output_array, int output_capacity) {
    // 计算当前线程的全局索引 input_index。如果索引超出范围或元素值小于阈值，则直接返回。
    int input_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (input_array[input_index] < threshold) { return; }

    /*
     * 使用原子操作 atomicAdd 递增并获取当前输出索引 output_index。
     * 如果输出索引超出容量，则返回。否则，存储满足条件的元素及其索引。
     */
    int output_index = atomicAdd(output_array, 1);
    if (output_index >= output_capacity) { return; }
    // 跳过第一个（第一个存的是元素数量），与前output_index * 2个元素
    float *output_item = output_array + 1 + output_index * 2;
    output_item[0] = input_array[input_index];
    output_item[1] = input_index;
}

void launch_keep_item(float *input_array, int input_size, float threshold, float *output_array, int output_capacity) {
    const int nthreads = 512;
    int block_size = input_size < nthreads ? input_size : nthreads;
    int gird_size = (input_size + block_size - 1) / block_size;

    launch_keep_item_kernel<<<gird_size, block_size, block_size * sizeof(float), nullptr>>>(input_array, input_size, threshold, output_array, output_capacity);
}
