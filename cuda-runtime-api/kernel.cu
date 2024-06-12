#include "cuda-runtime-api.h"

__global__ void test_print_kernel(const float *pdata, int ndata) {
    int idx = threadIdx.x;
    /*
     * blockIdx（线程块在线程网格内的位置索引）、threadIdx（线程在线程块内的位置索引）
     * 一个核函数只能有一个grid，一个grid可以有很多个block，每个block可以有很多的线程，
     * 网格和块的维度一般是二维和三维的，也就是说一个网格通常被分成二维的块，而每个块常被分成三维的线程。
     */
    printf("Element[%d] = %f, threadIdx.x=%d, blockIdx.x=%d, blockDim.x=%d\n", idx, pdata[idx], threadIdx.x, blockIdx.x, blockDim.x);
}

void test_print(const float *pdata, int ndata) {
    /*
     * 尖括号里面的参数类型：<<<gridDim, blockDim, bytes_of_shared_memory, stream>>>
     * gridDim 与 blockDim 都是 dim3 类型，dim3 的三个属性 int x y z 与shape类似
     */
    test_print_kernel<<<1, ndata, 0, nullptr>>>(pdata, ndata);

    /*
     * 在核函数执行结束后，通过cudaPeekAtLastError获取得到的代码，来知道是否出现错误
     * cudaPeekAtLastError和cudaGetLastError都可以获取得到错误代码
     * cudaGetLastError是获取错误代码并清除掉，也就是再一次执行cudaGetLastError获取的会是success
     * 而cudaPeekAtLastError是获取当前错误，但是再一次执行 cudaPeekAtLastError 或者 cudaGetLastError 拿到的还是那个错
     * cuda的错误会传递，如果这里出错了，不移除。那么后续的任意api的返回值都会是这个错误，都会失败
     */
    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess) {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("kernel error %s:%d  test_print_kernel failed. \n  code = %s, message = %s\n", __FILE__, __LINE__, err_name, err_message);
    }
}
