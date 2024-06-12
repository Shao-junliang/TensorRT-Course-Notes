#include "cuda-runtime-api.h"

// 函数宏 define 宏名称(参数1, 参数2, ...) (宏展开代码)
#define min(a, b) ((a) < (b) ? (a) : (b))

void cuda_runtime_api_9_atomic() {
    const int n = 100000;
    float *input_host = new float[n];
    float *input_device = nullptr;
    for (int i = 0; i < n; ++i) {
        // input_host[i] 的值是[0~99]的循环，循环了1000次
        input_host[i] = i % 100;
    }

    checkRuntime(cudaMalloc(&input_device, sizeof(float) * n));
    checkRuntime(cudaMemcpy(input_device, input_host, sizeof(float) * n, cudaMemcpyHostToDevice));

    // 分配并初始化输出数组 output_device。第一项存储输出元素的数量，剩余项存储满足条件的值和索引。
    int output_capacity = 20;
    float *output_host = new float[1 + output_capacity * 2];
    float *output_device = nullptr;
    checkRuntime(cudaMalloc(&output_device, (1 + output_capacity * 2) * sizeof(float)));
    checkRuntime(cudaMemset(output_device, 0, sizeof(float)));

    float threshold = 99;
    launch_keep_item(input_device, n, threshold, output_device, output_capacity);
    checkRuntime(cudaPeekAtLastError());

    checkRuntime(cudaMemcpy(output_host, output_device, (1 + output_capacity * 2) * sizeof(float), cudaMemcpyDeviceToHost));
    checkRuntime(cudaDeviceSynchronize());

    printf("output_size = %f\n", output_host[0]);
    int output_size = min(output_host[0], output_capacity);
    for (int i = 0; i < output_size; ++i) {
        float *output_item = output_host + 1 + i * 2;
        float value = output_item[0];
        int index = output_item[1];
        printf("output_host[%d] = %f, %d\n", i, value, index);
    }

    cudaFree(input_device);
    cudaFree(output_device);
    delete[] input_host;
    delete[] output_host;

    return;
}