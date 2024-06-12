#include "cuda-runtime-api.h"

void cuda_runtime_api_5_thread_layout() {
    cudaDeviceProp prop;
    checkRuntime(cudaGetDeviceProperties(&prop, 0));

    /*
     * 利用设备属性查询maxGridSize和maxThreadsDim参数，
     * 可以得到gridDims、blockDims的最大值，warpSize为线程束的线程数量。
     * maxThreadsPerBlock为一个block中能够容纳的最大线程数，
     * 即blockDims[0] * blockDims[1] * blockDims[2] <= maxThreadsPerBlock
     */
    printf("prop.maxGridSize = %d, %d, %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("prop.maxThreadsDim = %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("prop.warpSize = %d\n", prop.warpSize);
    printf("prop.maxThreadsPerBlock = %d\n", prop.maxThreadsPerBlock);

    int girds[] = {1, 2, 3};     // 该网格包含6个block
    int blocks[] = {1024, 1, 1}; // 每个block有1024个线程
    print_layout(girds, blocks); // 数组名本身就是指向第一个元素的指针，所以此处可以不用显示的取地址
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaDeviceSynchronize());
    printf("done\n");

    return;
}