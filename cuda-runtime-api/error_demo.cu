#include "cuda-runtime-api.h"

__global__ void func(float *ptr) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos == 999) {
        ptr[999] = 5;
    }
}

void error_demo() {
    float *ptr = nullptr;

    // 因为核函数是异步的，因此不会立即检查到是否存在异常，因为参数校验正常，所以返回 no error
    func<<<100, 30>>>(ptr);
    auto code1 = cudaPeekAtLastError();
    std::cout << cudaGetErrorString(code1) << std::endl; // no error

    // 对当前设备的核函数进行同步，等待执行完毕，可以发现过程是否存在异常
    auto code2 = cudaDeviceSynchronize();
    std::cout << cudaGetErrorString(code2) << std::endl; // an illegal memory access was encountered

    // 异常会一直存在，以至于后续的函数都会失败
    float *new_ptr = nullptr;
    auto code3 = cudaMalloc(&new_ptr, 100);
    std::cout << cudaGetErrorString(code3) << std::endl; // an illegal memory access was encountered

    return;
}