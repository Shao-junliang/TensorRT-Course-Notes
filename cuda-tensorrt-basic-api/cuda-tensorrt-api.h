// tensorRT include
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
// cuda include
#include <cuda_runtime.h>
// system include
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <functional>
#include <cassert>
#include "utils.h"

using namespace nvinfer1;

inline const char *severity_string(nvinfer1::ILogger::Severity t) {
    switch (t) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
    case nvinfer1::ILogger::Severity::kERROR: return "error";
    case nvinfer1::ILogger::Severity::kWARNING: return "warning";
    case nvinfer1::ILogger::Severity::kINFO: return "info";
    case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
    default: return "unknow";
    }
}

class TRTLogger : public nvinfer1::ILogger {
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override {
        if (severity <= Severity::kINFO) {
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if (severity == Severity::kWARNING) {
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            } else if (severity <= Severity::kERROR) {
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            } else {
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
};

void cuda_tensorrt_basic_api_1_builder();

void cuda_tensorrt_basic_api_2_inference();

void cuda_tensorrt_basic_api_3_dynamic_shape();

void cuda_tensorrt_basic_api_5_onnx_parser();

void cuda_tensorrt_basic_api_6_onnx_plugin();

void cuda_tensorrt_basic_api_7_integrate_easyplugin();

void cuda_tensorrt_basic_api_8_quantization();
