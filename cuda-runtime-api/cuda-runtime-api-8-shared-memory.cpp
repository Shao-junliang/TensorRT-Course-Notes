#include "cuda-runtime-api.h"

void cuda_runtime_api_8_shared_memory() {
    // 获取设备属性（此处为共享内存大小）
    cudaDeviceProp prop;
    checkRuntime(cudaGetDeviceProperties(&prop, 0));
    printf("prop.sharedMemPerBlock = %.2f KB\n", prop.sharedMemPerBlock / 1024.0f);

    launch();
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaDeviceSynchronize());

    return;
}