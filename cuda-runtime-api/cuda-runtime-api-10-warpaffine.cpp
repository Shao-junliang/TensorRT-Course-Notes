#include "cuda-runtime-api.h"

#define min(a, b) ((a) < (b) ? (a) : (b))

cv::Mat warpaffine_to_center_align(const cv::Mat &image, const cv::Size &size) {
    // 创建一个 size 尺寸的空图像
    cv::Mat output(size, CV_8UC3);

    uint8_t *psrc_device = nullptr;
    uint8_t *pdst_device = nullptr;
    size_t src_size = image.cols * image.rows * 3;
    size_t dst_size = size.width * size.height * 3;

    // gpu 上分配两块空间
    checkRuntime(cudaMalloc(&psrc_device, src_size));
    checkRuntime(cudaMalloc(&pdst_device, dst_size));

    // image 数据拷贝到 gpu
    checkRuntime(cudaMemcpy(psrc_device, image.data, src_size, cudaMemcpyHostToDevice));

    warp_affine_bilinear(
        psrc_device, image.cols * 3, image.cols, image.rows,
        pdst_device, size.width * 3, size.width, size.height,
        114);

    // 检查核函数执行是否存在错误
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaMemcpy(output.data, pdst_device, dst_size, cudaMemcpyDeviceToHost));
    checkRuntime(cudaFree(psrc_device));
    checkRuntime(cudaFree(pdst_device));
    return output;
}

void cuda_runtime_api_10_warpaffine() {
    cv::Mat img = cv::imread("../src/cuda-runtime-api/static/10.1.yq.jpg");
    cv::Mat output = warpaffine_to_center_align(img, cv::Size(640, 640));
    cv::imwrite("../src/cuda-runtime-api/static/10.1.output.jpg", output);
    printf("Done. save to output.jpg\n");
    return;
}