#include "cuda-runtime-api.h"

void cuda_runtime_api_7_add() {
    const int size = 3;
    // cpu 上的数据
    float vector_a[size] = {2, 3, 2};
    float vector_b[size] = {5, 3, 3};
    float vector_c[size] = {0};

    // gpu 上定义的内存地址
    float *vector_a_device = nullptr;
    float *vector_b_device = nullptr;
    float *vector_c_device = nullptr;
    // 分配内存
    checkRuntime(cudaMalloc(&vector_a_device, size * sizeof(float)));
    checkRuntime(cudaMalloc(&vector_b_device, size * sizeof(float)));
    checkRuntime(cudaMalloc(&vector_c_device, size * sizeof(float)));

    // 数据拷贝 cpu --> gpu
    checkRuntime(cudaMemcpy(vector_a_device, vector_a, size * sizeof(float), cudaMemcpyHostToDevice));
    checkRuntime(cudaMemcpy(vector_b_device, vector_b, size * sizeof(float), cudaMemcpyHostToDevice));

    vector_add(vector_a_device, vector_b_device, vector_c_device, size);
    // 结果从 gpu --> cpu
    checkRuntime(cudaMemcpy(vector_c, vector_c_device, size * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < size; ++i) {
        printf("vector_c[%d] = %f\n", i, vector_c[i]);
    }

    // 释放内存
    checkRuntime(cudaFree(vector_a_device));
    checkRuntime(cudaFree(vector_b_device));
    checkRuntime(cudaFree(vector_c_device));

    return;
}