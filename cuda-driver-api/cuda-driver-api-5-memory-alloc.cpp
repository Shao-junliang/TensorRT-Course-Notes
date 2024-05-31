#include "cuda-driver-api.h"
#include "utils.h"

void cuda_driver_api_5_memory_alloc() {
    // 检查cuda driver的初始化
    checkDriver(cuInit(0));

    // 创建上下文
    CUcontext context = nullptr;
    CUdevice device = 0;
    checkDriver(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
    printf("context = %p\n", context);

    // 输入device prt向设备要一个100 byte的线性内存，并返回地址
    CUdeviceptr device_memory_pointer = 0;
    // 注意这是指向device的pointer,
    checkDriver(cuMemAlloc(&device_memory_pointer, 100));
    printf("device_memory_pointer = %p\n", device_memory_pointer);

    /*
     * 输入二级指针向host要一个100 byte的锁页内存，专供设备访问。
     * 参考 static/5.2.cuMemAllocHost.jpg 讲解视频：https://v.douyin.com/NrYL5KB/
     * 页锁定内存是指不能交换到磁盘的内存
     * (void**)是将 float* 类型的指针转换为 void** 类型的指针
     */
    float *host_page_locked_memory = nullptr;
    checkDriver(cuMemAllocHost((void **)&host_page_locked_memory, 100));
    printf("host_page_locked_memory = %p\n", host_page_locked_memory);

    // 向page-locked memory 里放数据（仍在CPU上），可以让GPU可快速读取
    host_page_locked_memory[0] = 123;
    printf("host_page_locked_memory[0] = %f\n", host_page_locked_memory[0]);

    /*
     * 记住这一点
     * host page locked memory 声明的时候为float*型，可以直接转换为device ptr，这才可以送给cuda核函数（利用DMA(Direct Memory Access)技术）
     * 初始化内存的值: cuMemsetD32(CUdeviceptr dstDevice, unsigned int  ui, size_t N)
     * 初始化值必须是无符号整型，因此需要将new_value进行数据转换：
     * 但不能直接写为:(int)value，必须写为*(int*)&new_value, 我们来分解一下这条语句的作用：
     * 1. &new_value获取float new_value的地址
     * (int*)将地址从float * 转换为int*以避免64位架构上的精度损失
     * *(int*)取消引用地址，最后获取引用的int值
     */
    float new_value = 555;
    checkDriver(cuMemsetD32((CUdeviceptr)host_page_locked_memory, *(int *)&new_value, 1));
    printf("host_page_locked_memory[0] = %f\n", host_page_locked_memory[0]);

    // 释放内存
    checkDriver(cuMemFreeHost(host_page_locked_memory));
    return;
}