#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <cstdio>

// 宏定义: #define <宏名>(<参数表>) <宏体>
#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line);

struct Box {
    float left, top, right, bottom, confidence;
    int label;

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int label) :
        left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label) {
    }
};

#endif // UTILS_H
