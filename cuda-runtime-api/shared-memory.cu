#include "cuda-runtime-api.h"
#include <vcruntime.h>

// 设置内存大小
const size_t static_shared_memory_num_element = 6 * 1024; // 6KB
// 定义静态共享内存 memory、memory2
__shared__ char static_shared_memory[static_shared_memory_num_element];
__shared__ char static_shared_memory2[2];

__global__ void demo1_kernel() {
    // 定义动态共享内存，静态共享变量和动态共享变量在kernel函数内/外定义都行，没有限制
    extern __shared__ char dynamic_shared_memory[];
    extern __shared__ char dynamic_shared_memory2[];

    // 静态共享变量，定义几个地址随之叠加
    printf("static_shared_memory = %p\n", static_shared_memory);
    printf("static_shared_memory2 = %p\n", static_shared_memory2);

    // 动态共享变量，无论定义多少个，地址都一样
    printf("dynamic_shared_memory = %p\n", dynamic_shared_memory);
    printf("dynamic_shared_memory2 = %p\n", dynamic_shared_memory2);

    // 第一个thread
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Run kernel.\n");
    }
}

__shared__ int shared_value1;

__global__ void demo2_kernel() {
    __shared__ int shared_value2;
    if (blockIdx.x == 0) {
        shared_value1 = 666;
        shared_value2 = 66;
    } else {
        shared_value1 = 888;
        shared_value2 = 88;
    }

    // __syncthreads() 用来同步block内的所有线程，全部线程执行到这一行时才会往下继续执行
    __syncthreads();

    printf("%d.%d. shared_value1 = %d[%p], shared_value2 = %d[%p]\n", blockIdx.x,
           threadIdx.x, shared_value1, &shared_value1, shared_value2,
           &shared_value2);
}

void launch() {
    // demo1 主要为了展示查看静态和动态共享变量的地址
    demo1_kernel<<<1, 1, 12, nullptr>>>();
    // demo2 主要是为了演示的是如何给共享变量进行赋值
    demo2_kernel<<<2, 5, 0, nullptr>>>();
}