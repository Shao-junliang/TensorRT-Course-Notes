#include "cuda-runtime-api.h"

__global__ void gemm_kernel_0(const float *A, const float *B, float *C, int m, int n, int k) {
    /*
     * 该方法实现了通用矩阵乘法，计算了两个矩阵 𝐴 和 𝐵 的乘积，并将结果存储在矩阵 𝐶 中。
     * A: m x n
     * B: n x k
     * C: m x k
     */

    // row,col 是目标矩阵的坐标；
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 角标越界
    if (row >= m || col >= k) { return; }

    float csum = 0;
    for (int i = 0; i < n; ++i) {
        csum += A[row * n + i] * B[i * k + col];
    }
    C[row * k + col] = csum;
}

void gemm_0(const float *A, const float *B, float *C, int m, int n, int k, cudaStream_t stream) {
    dim3 block(16, 16, 1);
    dim3 grid((k + block.x - 1) / block.x, (m + block.y - 1) / block.y, 1);
    gemm_kernel_0<<<grid, block, 0, stream>>>(A, B, C, m, n, k);
}

// BLOCK_SIZE 是一个模板参数，用于指定线程块的大小。这个参数在编译时确定，用于定义共享内存和线程块的尺寸。
template <int BLOCK_SIZE>
__global__ void gemm_kernel_1(const float *A, const float *B, float *C, int m, int n, int k) {
    // 定义静态共享内存，二维矩阵，用于存储当前块中 A 和 B 的子矩阵。
    __shared__ float shardM[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shardN[BLOCK_SIZE][BLOCK_SIZE];

    // bx, by 是当前块的 x 和 y 索引。
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // tx, ty 是当前线程在块中的 x 和 y 索引。
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // row, col 是当前线程在全局矩阵中的位置。
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    float v = 0.0;

    // ceil 会返回大于或等于该值的最小整数。类似于 "进一法"
    for (int i = 0; i < (int)(ceil((float)n / BLOCK_SIZE)); i++) {
        // 加载 A 的子矩阵到共享内存
        if (i * BLOCK_SIZE + tx < n && row < m) {
            shardM[ty][tx] = A[row * n + i * BLOCK_SIZE + tx];
        } else {
            shardM[ty][tx] = 0.0;
        }
        // 加载 B 的子矩阵到共享内存
        if (i * BLOCK_SIZE + ty < n && col < k) {
            shardN[ty][tx] = B[(i * BLOCK_SIZE + ty) * k + col];
        } else {
            shardN[ty][tx] = 0.0;
        }

        // 同步线程 使线程块中的所有线程等待，确保共享内存加载完成。
        __syncthreads();

        // 行列相乘，计算出某个位置的值
        for (int j = 0; j < BLOCK_SIZE; j++) {
            v += shardM[ty][j] * shardN[j][tx];
        }
        __syncthreads();
    }
    // 赋值
    if (row < m && col < k) { C[row * k + col] = v; }
}

void gemm_1(const float *A, const float *B, float *C, int m, int n, int k, cudaStream_t stream) {
    dim3 block(16, 16, 1);
    dim3 grid((k + block.x - 1) / block.x, (m + block.y - 1) / block.y, 1);
    // 在此处指定了 BLOCK_SIZE 的值，值为 16
    gemm_kernel_1<16><<<grid, block, 0, stream>>>(A, B, C, m, n, k);
}