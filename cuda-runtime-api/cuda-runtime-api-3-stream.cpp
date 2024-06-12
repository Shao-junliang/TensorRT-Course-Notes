#include "cuda-runtime-api.h"

void cuda_runtime_api_3_stream() {
    // 设置运行设备
    int device_id = 0;
    checkRuntime(cudaSetDevice(device_id));

    // 创建流
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    // 在GPU上开辟空间
    float *memory_device = nullptr;
    checkRuntime(cudaMalloc(&memory_device, 100 * sizeof(float)));

    // 在cpu上开辟空间，并且存放数据，并将cpu上的数据复制到gpu上去
    float *memory_host = new float[100];
    memory_host[2] = 520.25;
    // 注意：此处是异步操作，会立即返回，所以要考虑传入参数的生命周期，切不可在同步操作之前结束传参的生命周期；
    checkRuntime(cudaMemcpyAsync(memory_device, memory_host, 100 * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));

    // 在cpu上开辟 pin memory，并将gpu上的数据复制到 pin memory
    float *memory_page_locked = nullptr;
    checkRuntime(cudaMallocHost(&memory_page_locked, 100 * sizeof(float)));
    checkRuntime(cudaMemcpyAsync(memory_page_locked, memory_device,
                                 100 * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // 同步操作
    checkRuntime(cudaStreamSynchronize(stream));

    // 结果打印
    printf("%f\n", memory_page_locked[2]);

    // 释放内存
    checkRuntime(cudaFreeHost(memory_page_locked));
    checkRuntime(cudaFree(memory_device));
    checkRuntime(cudaStreamDestroy(stream));
    delete[] memory_host;

    return;
}