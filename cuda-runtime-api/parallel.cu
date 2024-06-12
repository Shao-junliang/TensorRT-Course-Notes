#include "cuda-runtime-api.h"
#include "utils.h"

__global__ void add_vector(const float *a, const float *b, float *c, int count) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= count) { return; }
    c[index] = a[index] + b[index];
}

__global__ void mul_vector(const float *a, const float *b, float *c, int count) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= count) { return; }
    c[index] = a[index] * b[index];
}

cudaStream_t stream1, stream2;
float *a, *b, *c1, *c2;
const int num_element = 100000;
const size_t bytes = sizeof(float) * num_element;
const int blocks = 512;
const int girds = (num_element + blocks - 1) / blocks;
const int ntry = 1000;

void async() {
    cudaEvent_t event_start1, event_stop1;
    cudaEvent_t event_start2, event_stop2;
    // 创建事件
    checkRuntime(cudaEventCreate(&event_start1));
    checkRuntime(cudaEventCreate(&event_stop1));
    checkRuntime(cudaEventCreate(&event_start2));
    checkRuntime(cudaEventCreate(&event_stop2));

    auto tic = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    // 记录事件开始时间，在 stream1 中加入 event_start1
    checkRuntime(cudaEventRecord(event_start1, stream1));
    for (int i = 0; i < ntry; i++) {
        add_vector<<<girds, blocks, 0, stream1>>>(a, b, c1, num_element);
    }
    // 记录事件结束时间
    checkRuntime(cudaEventRecord(event_stop1, stream1));

    checkRuntime(cudaEventRecord(event_start2, stream2));
    for (int i = 0; i < ntry; i++) {
        add_vector<<<girds, blocks, 0, stream2>>>(a, b, c2, num_element);
    }
    checkRuntime(cudaEventRecord(event_stop2, stream2));

    // 等待流里面的指令全部都执行完，同步操作
    checkRuntime(cudaStreamSynchronize(stream1));
    checkRuntime(cudaStreamSynchronize(stream2));
    auto toc = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;

    float time1, time2;
    // 计算两个事件之间经历的时间间隔
    checkRuntime(cudaEventElapsedTime(&time1, event_start1, event_stop1));
    checkRuntime(cudaEventElapsedTime(&time2, event_start2, event_stop2));
    printf("async: time1 = %.2f ms, time2 = %.2f ms, count = %.2f ms\n", time1, time2, toc - tic);
}

void sync() {
    cudaEvent_t event_start1, event_stop1;
    checkRuntime(cudaEventCreate(&event_start1));
    checkRuntime(cudaEventCreate(&event_stop1));

    auto tic = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    checkRuntime(cudaEventRecord(event_start1, stream1));
    for (int i = 0; i < ntry; i++) {
        add_vector<<<girds, blocks, 0, stream1>>>(a, b, c1, num_element);
    }
    for (int i = 0; i < ntry; i++) {
        add_vector<<<girds, blocks, 0, stream2>>>(a, b, c2, num_element);
    }
    checkRuntime(cudaEventRecord(event_stop1, stream1));

    checkRuntime(cudaStreamSynchronize(stream1));
    auto toc = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;

    float time1;
    checkRuntime(cudaEventElapsedTime(&time1, event_start1, event_stop1));
    printf("sync: time1 = %.2f ms, count = %.2f ms\n", time1, toc - tic);
}

void multi_stream_async() {
// #define <宏名称> <宏定义的替换文本>
#define step1 add_vector
#define step2 mul_vector
#define step3 add_vector
#define step4 mul_vector
#define stepa add_vector

    /*
     * 创建了一个事件，
     * 先在 stream1 中依次执行 step1 -> step2。
     * 然后在 stream2 中执行 stepa。
     * 最后在 stream1 中执行 step3 -> step4，在 step3 开始之前等待 stepa 结束的事件。
     * step3 需要 step2 的输出：因为 step3 是在 step1 和 step2 之后执行的，在同一个 stream1 中，所有前面的操作都会按照顺序完成。
     * step3 需要 stepa 的输出：通过 cudaStreamWaitEvent，确保 step3 的执行是在 stepa 完成之后。stream1 在等待 event_async 事件之前会暂停，直到 stepa 完成并触发事件 event_async。
     */

    cudaEvent_t event_async;
    checkRuntime(cudaEventCreate(&event_async));

    // 依次执行 step1 --> step2
    step1<<<girds, blocks, 0, stream1>>>(a, b, c1, num_element);
    step2<<<girds, blocks, 0, stream1>>>(a, b, c1, num_element);

    // 等待流中的某个事件，此处事件为 event_async
    checkRuntime(cudaStreamWaitEvent(stream1, event_async));

    // 异步操作，此处只是进行了并直接返回，但还没执行，因为 stream1 处于等待状态，
    step3<<<girds, blocks, 0, stream1>>>(a, b, c2, num_element);
    step4<<<girds, blocks, 0, stream1>>>(a, b, c2, num_element);

    // 执行 stepa
    stepa<<<girds, blocks, 0, stream2>>>(a, b, c2, num_element);

    // 记录事件；相当于触发了 event_async 事件，上面的 stream1 不再等待，开始执行 step3 --> step4
    checkRuntime(cudaEventRecord(event_async, stream2));

    // 同步操作
    checkRuntime(cudaStreamSynchronize(stream1));

    printf("multi_stream_async done.\n");
}

void parallel() {
    // 创建两个流
    checkRuntime(cudaStreamCreate(&stream1));
    checkRuntime(cudaStreamCreate(&stream2));

    // 分配GPU内存
    checkRuntime(cudaMalloc(&a, bytes));
    checkRuntime(cudaMalloc(&b, bytes));
    checkRuntime(cudaMalloc(&c1, bytes));
    checkRuntime(cudaMalloc(&c2, bytes));

    async(); // 两个流异步执行

    sync(); // 单个流串行

    multi_stream_async(); // 多个流之间并行
}
