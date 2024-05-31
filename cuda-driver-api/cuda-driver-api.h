#ifndef CUDA_DRIVER_API_H
#define CUDA_DRIVER_API_H

/*
 * include <> 和 "" 的区别
 * include <> : 标准库文件
 * include "" : 自定义文件
 */
#include <cuda.h>
#include <stdio.h> // 因为要使用printf
#include <string.h>

void cuda_driver_api_1_cuinit();

void cuda_driver_api_2_check1();

void cuda_driver_api_3_check2();

void cuda_driver_api_4_context();

void cuda_driver_api_5_memory_alloc();

#endif // CUDA_DRIVER_API_H
