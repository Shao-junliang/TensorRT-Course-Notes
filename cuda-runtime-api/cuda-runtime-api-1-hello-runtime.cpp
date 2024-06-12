#include "cuda-runtime-api.h"

void cuda_runtime_api_1_hello_runtime() {
    /*
     * cuda runtime 是以 cuda 为基准开发的运行时库。
     * cuda runtime 所使用的 CUcontext 是基于 cuDevicePrimaryCtxRetain 函数获取的。
     * cuDevicePrimaryCtxRetain 会为每个设备关联一个context，通过 cuDevicePrimaryCtxRetain 函数可以获取到。
     * 而 context 初始化的时机是懒加载模式，当调用一个 runtime api 时，会自动触发创建动作。
     * 也因此，避免了cu驱动级别的 init 和 destroy 操作。使得api的调用更加容易。
     */
    CUcontext context = nullptr;
    cuCtxGetCurrent(&context);
    printf("Current context = %p, no context\n", context);

    // 检查显卡的个数
    int device_count = 0;
    checkRuntime(cudaGetDeviceCount(&device_count));
    printf("device_count = %d\n", device_count);

    /*
     * 使用setdevice来控制当前上下文，当要使用不同设备时，使用不同的 device id
     * 注意，context是线程内作用的，其他线程不相关的, 一个线程一个 context stack
     */
    int device_id = 0;
    printf("set current device to : %d, this API rely on CUcontext, emit and setting \n", device_id);
    checkRuntime(cudaSetDevice(device_id));

    /*
     * 注意：由于cudaSetDevice函数是"第一个执行的需要context的函数"，
     * 所以会执行cuDevicePrimaryCtxRetain，并设置当前context，这一切都是默认执行的。
     * 注意：cudaGetDeviceCount是一个不需要context的函数，你可以认为绝大部分runtime api都是需要context的，
     * 所以第一个执行的cuda runtime函数，会创建context并设置上下文。
     */
    cuCtxGetCurrent(&context);
    printf("SetDevice after, Current context = %p, get current context\n", context);

    int current_device = 0;
    checkRuntime(cudaGetDevice(&current_device));
    printf("current_device = %d\n", current_device);
    return;
}