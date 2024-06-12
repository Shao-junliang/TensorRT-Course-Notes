#ifndef CUDA_RUNTIME_API_H
#define CUDA_RUNTIME_API_H

// CUDA运行时头文件
#include <cuda_runtime.h>
#include <cublas_v2.h>
// CUDA驱动头文件
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "utils.h"

void cuda_runtime_api_1_hello_runtime();

void cuda_runtime_api_2_memory();

void cuda_runtime_api_3_stream();

void cuda_runtime_api_4_kernel_function();

void cuda_runtime_api_5_thread_layout();

void cuda_runtime_api_6_parallel();

void cuda_runtime_api_7_add();

void cuda_runtime_api_8_shared_memory();

void cuda_runtime_api_9_atomic();

void cuda_runtime_api_10_warpaffine();

void cuda_runtime_api_11_cublas_gemm();

void cuda_runtime_api_12_yolov5_postprocess();

void cuda_runtime_api_13_thrust();

void cuda_runtime_api_14_error();

void test_print(const float *pdata, int ndata); // 4.cpp

void print_layout(int *girds, int *blocks); // 5.cpp

void parallel(); // 6.cpp

void vector_add(const float *a, const float *b, float *c, int ndata); // 7.cpp

void launch(); // 8.cpp

void launch_keep_item(float *input_array, int input_size,
                      float threshold, float *output_array, int output_capacity); // 9.cpp

void warp_affine_bilinear(
    uint8_t *src, int src_line_size, int src_width, int src_height,
    uint8_t *dst, int dst_line_size, int dst_width, int dst_height,
    uint8_t fill_value); // 10.cpp

void gemm_0(const float *A, const float *B, float *C,
            int m, int n, int k, cudaStream_t stream); // 11.cpp

void gemm_1(const float *A, const float *B, float *C,
            int m, int n, int k, cudaStream_t stream); // 11.cpp

std::vector<uint8_t> load_file(const std::string &file); // 12,cpp

std::vector<Box> cpu_decode(float *predict, int rows, int cols, float confidence_threshold = 0.25f, float nms_threshold = 0.45f); // 12,cpp

std::vector<Box> gpu_decode(float *predict, int rows, int cols, float confidence_threshold = 0.25f, float nms_threshold = 0.45f); // 12,cpp

void decode_kernel_invoker(
    float *predict, int num_bboxes, int num_classes, float confidence_threshold,
    float nms_threshold, float *invert_affine_matrix, float *parray, int max_objects,
    int NUM_BOX_ELEMENT, cudaStream_t stream); // 12.cpp

void thrust_demo(); // 13.cpp

void error_demo(); // 14.cpp

#endif // CUDA_RUNTIME_API_H
