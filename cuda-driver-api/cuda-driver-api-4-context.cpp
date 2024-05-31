#include "cuda-driver-api.h"
#include "utils.h"

void cuda_driver_api_4_context() {
    // 检查cuda driver的初始化
    checkDriver(cuInit(0));

    // 为设备创建上下文
    CUcontext ctxA = nullptr; // CUcontext 其实是 struct CUctx_st*（是一个指向结构体CUctx_st的指针）
    CUcontext ctxB = nullptr;
    CUdevice device = 0;

    // 这一步相当于告知要某一块设备上的某块地方创建 ctxA 管理数据。
    checkDriver(cuCtxCreate(&ctxA, CU_CTX_SCHED_AUTO, device));
    // 参考 static/4.1.ctx-stack.jpg
    checkDriver(cuCtxCreate(&ctxB, CU_CTX_SCHED_AUTO, device));
    /*
        contexts 栈：
            ctxB -- top <--- current_context
            ctxA
            ...
     */
    printf("ctxA = %p\n", ctxA);
    printf("ctxB = %p\n", ctxB);

    // 获取当前上下文信息
    CUcontext current_context = nullptr;
    checkDriver(cuCtxGetCurrent(&current_context));
    // 这个时候current_context 栈顶的 ctxB
    printf("current_context = %p\n", current_context);

    // 可以使用上下文堆栈对设备管理多个上下文
    // 压入当前context，将 ctxA 压入CPU调用的thread上。专门用一个thread以栈的方式来管理多个contexts的切换
    checkDriver(cuCtxPushCurrent(ctxA));
    // 获取current_context (即栈顶的context)
    checkDriver(cuCtxGetCurrent(&current_context));
    /*
        contexts 栈：
            ctxA -- top <--- current_context
            ctxB
            ...
    */
    printf("after pushing, current_context = %p\n", current_context);

    // 弹出当前context
    CUcontext popped_ctx = nullptr;
    // 将当前的context pop掉，并用popped_ctx承接它pop出来的context
    checkDriver(cuCtxPopCurrent(&popped_ctx));
    // 获取current_context(栈顶的)
    checkDriver(cuCtxGetCurrent(&current_context));
    // 弹出的是ctxA
    printf("after poping, popped_ctx = %p\n", popped_ctx);
    // current_context是ctxB
    printf("after poping, current_context = %p\n", current_context);

    checkDriver(cuCtxDestroy(ctxA));
    checkDriver(cuCtxDestroy(ctxB));

    // 更推荐使用cuDevicePrimaryCtxRetain获取与设备关联的context
    // 注意这个重点，以后的runtime也是基于此, 自动为设备只关联一个context
    // 在 device 上指定一个新地址对ctxA进行管理
    checkDriver(cuDevicePrimaryCtxRetain(&ctxA, device));
    printf("ctxA = %p\n", ctxA);
    checkDriver(cuDevicePrimaryCtxRelease(device));
    return;
}
