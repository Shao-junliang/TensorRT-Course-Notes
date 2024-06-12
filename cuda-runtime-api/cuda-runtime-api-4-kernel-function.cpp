#include "cuda-runtime-api.h"

void cuda_runtime_api_4_kernel_function() {
    // 创建cpu内存与gpu内存的首地址指针
    float *parray_host = nullptr;
    float *parray_device = nullptr;
    int narray = 10; // 存储空间大小
    int array_bytes = sizeof(float) * narray;

    // cpu上开辟空间
    parray_host = new float[narray];
    checkRuntime(cudaMalloc(&parray_device, array_bytes));

    // cpu上每个空间复制
    for (int i = 0; i < narray; i++) {
        parray_host[i] = i;
    }

    // cpu上的值拷贝到gpu上去
    checkRuntime(cudaMemcpy(parray_device, parray_host, array_bytes, cudaMemcpyHostToDevice));
    test_print(parray_device, narray);
    // 同步操作
    checkRuntime(cudaDeviceSynchronize());

    // 释放内存
    checkRuntime(cudaFree(parray_device));
    delete[] parray_host;

    return;
}