#ifndef UTILS_H
#define UTILS_H

#include <cuda.h>
#include <cstdio>

// 宏定义: #define <宏名>(<参数表>) <宏体>
#define checkDriver(op) __check_cuda_driver((op), #op, __FILE__, __LINE__)

bool __check_cuda_driver(CUresult code, const char *op, const char *file, int line);

#endif // UTILS_H
