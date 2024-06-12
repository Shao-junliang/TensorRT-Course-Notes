#include "cuda-runtime-api.h"

#define min(a, b) ((a) < (b) ? (a) : (b))

// 用于验证两个矩阵之间的元素是否在给定的误差范围 eps 内相等。
void verify(const cv::Mat &a, const cv::Mat &b, float eps = 1e-5) {
    // 计算矩阵 a 和 b 之间的差异，并将结果存储在 diff 矩阵中。cv::Mat 的减法操作符会逐元素相减。
    auto diff = cv::Mat(a - b);

    // 获取 diff 矩阵中第一行的指针，以便后续通过指针遍历矩阵的每个元素。
    float *p = diff.ptr<float>(0);

    // 初始化错误计数器 error_count 为 0，以及最大和最小差异值 max_diff 和 min_diff 为 diff 矩阵第一个元素的值。
    int error_count = 0;
    float max_diff = *p;
    float min_diff = *p;

    // 遍历 diff 矩阵的每个元素。如果某个元素的绝对值大于等于误差范围 eps，
    for (int i = 0; i < diff.rows * diff.cols; ++i, ++p) {
        // fabs 是一个函数，用于计算浮点数的绝对值。
        if (fabs(*p) >= eps) {
            if (error_count < 10) { printf("Error value: %f, %d\n", *p, i); }
            error_count += 1;
        }
        max_diff = std::max(max_diff, *p);
        min_diff = min(min_diff, *p);
    }
    if (error_count > 0) {
        printf("... error count = %d. max = %f, min = %f\n", error_count, max_diff, min_diff);
    } else {
        printf("Done no error. max = %f, min = %f\n", max_diff, min_diff);
    }
}

void cuda_runtime_api_11_cublas_gemm() {
    cv::Mat A, B, C;
    int m = 2000, n = 3000, k = 5000;
    A.create(m, n, CV_32F);
    B.create(n, k, CV_32F);
    C.create(m, k, CV_32F);

    // cv::randn 用于生成具有正态（高斯）分布的随机数并填充给定的数组或矩阵。
    cv::randn(A, 0, 1);
    cv::randn(B, 0, 1);

    cudaEvent_t tc0, tc1, tc2, tc3;
    cudaStream_t stream = nullptr;

    cublasHandle_t cublas_h = nullptr;

    cudaStreamCreate(&stream);
    cublasCreate(&cublas_h);
    cublasSetStream(cublas_h, stream);
    cudaEventCreate(&tc0);
    cudaEventCreate(&tc1);
    cudaEventCreate(&tc2);
    cudaEventCreate(&tc3);

    int Abytes = m * n * sizeof(float);
    int Bbytes = k * n * sizeof(float);
    int Cbytes = m * k * sizeof(float);

    // cudaHostRegister 用于注册主机内存，使其成为可通过设备直接访问的页锁定内存。
    cudaHostRegister(A.data, Abytes, cudaHostRegisterDefault);
    cudaHostRegister(B.data, Bbytes, cudaHostRegisterDefault);
    cudaHostRegister(C.data, Cbytes, cudaHostRegisterDefault);

    // 分配 gpu 内存
    float *dA, *dB, *dC;
    cudaMalloc(&dA, Abytes);
    cudaMalloc(&dB, Bbytes);
    cudaMalloc(&dC, Cbytes);

    // 使用 cudaMemcpyAsync 函数将主机内存中的数据异步传输到设备内存中，通过 stream 执行，使得数据传输和计算可以重叠，提高效率。
    cudaMemcpyAsync(dA, A.data, Abytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dB, B.data, Bbytes, cudaMemcpyHostToDevice, stream);

    // 计算了 10 次 𝐶=𝐴×𝐵，主要作用是预热（warm-up）。预热步骤在 GPU 编程中非常常见，其目的是确保 GPU 处于最佳状态，以便后续的性能测量更加准确和可靠。
    for (int i = 0; i < 10; ++i) {
        // dA, dB, dC 分别是矩阵 A, B, C 在 GPU 上的指针，m, n, k 是矩阵的维度，stream 是 CUDA 的流。
        gemm_0(dA, dB, dC, m, n, k, stream);
    }

    // 记录第一次 gemm_0 调用开始的时间
    cudaEventRecord(tc0, stream);
    gemm_0(dA, dB, dC, m, n, k, stream);

    // 记录第一次 gemm_0 调用结束和 gemm_1 调用开始的时间
    cudaEventRecord(tc1, stream);
    gemm_1(dA, dB, dC, m, n, k, stream);

    // 记录第一次 gemm_1 调用结束和 cuBLAS 调用开始的时间
    cudaEventRecord(tc2, stream);

    float alpha = 1.0f;
    float beta = 0.0f;
    int lda = n;
    int ldb = k;
    int ldc = m;
    // 这个函数用来实现矩阵乘法，详情查阅：https://docs.nvidia.com/cuda/archive/11.7.0/cublas/index.html#cublas-lt-t-gt-gemm
    cublasSgemm(cublas_h,
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_T,
                m,
                k,
                n,
                &alpha,
                dA,
                lda,
                dB,
                ldb,
                &beta,
                dC,
                ldc);

    // 记录 cuBLAS 调用结束的时间
    cudaEventRecord(tc3, stream);

    // 计算结果从gpu复制到cpu，并同步CUDA 流以确保所有操作完成。
    cudaMemcpyAsync(C.data, dC, Cbytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 计算并打印 gemm_0、gemm_1 和 cuBLAS 矩阵乘法的执行时间。
    float time_kernel_0 = 0;
    float time_kernel_1 = 0;
    float time_kernel_cublas = 0;
    cudaEventElapsedTime(&time_kernel_0, tc0, tc1);
    cudaEventElapsedTime(&time_kernel_1, tc1, tc2);
    cudaEventElapsedTime(&time_kernel_cublas, tc2, tc3);

    printf("kernel_0 = %.5f ms, kernel_1 = %.5f ms, kernel_cublas = %.5f ms\n",
           time_kernel_0, time_kernel_1, time_kernel_cublas);

    // CPU 上的矩阵乘法和时间计算：
    auto t0 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    auto tC = cv::Mat(A * B);
    auto t1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;

    printf("CPU times: %.5f ms\n", t1 - t0);

    // 验证 GPU 计算结果是否正确。verify 函数用于比较两个矩阵的结果是否在给定误差范围内一致。
    bool cublas_result = true;
    if (cublas_result) {
        cv::Mat tmp(C.cols, C.rows, CV_32F, C.data);
        tmp = tmp.t();
        verify(tmp, tC, 1e-3);
    } else {
        verify(C, tC, 1e-3);
    }

    return;
}
