#include "utils.h"

bool __check_cuda_driver(CUresult code, const char *op, const char *file, int line) {
    if (code != CUresult::CUDA_SUCCESS) { // 如果 成功获取CUDA情况下的返回值 与我们给定的值(0)不相等， 即条件成立， 返回值为flase
        const char *err_name = nullptr;   // 定义了一个字符串常量的空指针
        const char *err_message = nullptr;
        cuGetErrorName(code, &err_name);
        cuGetErrorString(code, &err_message);
        printf("%s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}