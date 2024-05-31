#include "cuda-driver-api.h"
#include "utils.h"

void cuda_driver_api_3_check2() {
    // 检查cuda driver的初始化
    // 实际调用的是__check_cuda_driver这个函数
    checkDriver(cuInit(0));

    // 测试获取当前cuda驱动的版本
    int driver_version = 0;
    if (!checkDriver(cuDriverGetVersion(&driver_version))) {
        return;
    }
    printf("Driver version is %d\n", driver_version);

    // 测试获取当前设备信息
    char device_name[100];
    CUdevice device = 0;
    checkDriver(cuDeviceGetName(device_name, sizeof(device_name), device));
    printf("Device %d name is %s\n", device, device_name);

    return;
}