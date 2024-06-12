#include "cuda-runtime-api.h"

void cuda_runtime_api_2_memory() {
    int device_id = 0;
    checkRuntime(cudaSetDevice(device_id));

    // global memory
    float *memory_device = nullptr;
    // pointer to device
    checkRuntime(cudaMalloc(&memory_device, 100 * sizeof(float)));

    // pageable memory
    float *memory_host = new float[100];
    memory_host[2] = 520.25;
    // 返回的地址是开辟的device地址，存放在 memory_device
    checkRuntime(cudaMemcpy(memory_device, memory_host, 100 * sizeof(float), cudaMemcpyHostToDevice));

    // pinned memory == page locked memory
    float *memory_page_locked = nullptr;
    // 返回的地址是被开辟的pin memory的地址，存放在 memory_page_locked
    checkRuntime(cudaMallocHost(&memory_page_locked, 100 * sizeof(float)));
    checkRuntime(cudaMemcpy(memory_page_locked, memory_device, 100 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("%f\n", memory_page_locked[2]);

    // 释放内存；
    checkRuntime(cudaFreeHost(memory_page_locked));
    delete[] memory_host;
    checkRuntime(cudaFree(memory_device));

    return;
}