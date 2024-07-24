#include "cuda-tensorrt-api.h"
/*
 * onnx解析器的头文件
 * 1. 如果 NvOnnxParser.h 在 CUDA\v11.7\include 中，则为已编译好的so解析器
 * 2. 如果用源代码编译，则直接 include 源代码中的 NvOnnxParser.h
 * 本例因为是自己手写的op算子，所以用源代码解析
 */
// #include <NvOnnxParser.h>
#include "../../../3rd_third/onnx-tensorrt/NvOnnxParser.h"

bool build_customize_plugin_model() {
    TRTLogger logger;
    // --------------------------------- 1. 定义 builder、config、network 指针 ----------------------------------
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);

    // --------------------------------- 2. 模型结构与输入输出信息 ----------------------------------
    // 输入输出的数量与网络权重的初始化，通过onnxparser解析器将模型的权重填充到network中，类似addConv的方式添加进去
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile("../src/cuda-tensorrt-basic-api/static/plugin_demo.onnx", 1)) {
        printf("Failed to parser demo.onnx\n");

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return false;
    }

    // 超参初始化
    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    // 配置暂存存储器，用于layer实现的临时存储，也用于保存中间激活值
    config->setMaxWorkspaceSize(1 << 28);

    /* --------------------------------- 2.1 关于profile ----------------------------------
     * 如果模型有多个输入，则必须多个profile
     * 优化配置文件（Optimization Profile）允许你为动态输入张量设置最小（kMIN）、最优（kOPT）和最大（kMAX）维度。
     */
    auto profile = builder->createOptimizationProfile();
    auto input = network->getInput(0);
    auto num_input = input->getDimensions().d[1];

    // 设置动态维度的最小值
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, num_input, 3, 3));
    // 设置动态维度的最优值
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, num_input, 5, 5));
    // 设置动态维度的最大值
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, num_input, 5, 5));
    config->addOptimizationProfile(profile);
    // 生成engine文件
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr) {
        printf("Build engine failed.\n");
        return false;
    }

    // --------------------------------- 3. 序列化+保存 ----------------------------------
    nvinfer1::IHostMemory *model_data = engine->serialize();
    FILE *f = fopen("../src/cuda-tensorrt-basic-api/static/plugin_demo.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);
    // 内存释放
    model_data->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done.\n");
    return true;
}

void cuda_tensorrt_basic_api_6_onnx_plugin() {
    if (!build_customize_plugin_model()) {
        return;
    }
    TRTLogger logger;
    // 1.加载模型
    auto engine_data = CTA::load_file("../src/cuda-tensorrt-basic-api/static/plugin_demo.trtmodel");
    // 2.创建 runtime
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    // 3.反序列化得到 engine
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (engine == nullptr) {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }
    // 4.从 engine 创建上下文
    IExecutionContext *execution_context = engine->createExecutionContext();
    // 5.创建 CUDA 流
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    // 6.数据准备
    // 6.1 输入数据x初始化
    float input_data_host[] = {
        // batch 0
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,

        // batch 1
        -1, 1, 1,
        1, 0, 1,
        1, 1, -1};
    float *input_data_device = nullptr;

    // 6.2 声明 output 数组大小
    const int ib = 2;
    // const int ic = 1;
    const int iw = 3;
    const int ih = 3;
    float output_data_host[ib * iw * ih];
    float *output_data_device = nullptr;

    // 6.3 输入数据搬到 gpu 上
    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));
    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);

    // 6.4 因为是动态输入，所以在模型推理之前，需明确输入数据的大小
    execution_context->setBindingDimensions(0, nvinfer1::Dims4(ib, 1, iw, ih));
    // 6.5 构建 bindings 数组
    float *bindings[] = {input_data_device, output_data_device};
    // 6.6 模型推理
    bool success = execution_context->enqueueV2((void **)bindings, stream, nullptr);
    // 6.7 数据搬回到 cpu
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    // 7.流同步
    cudaStreamSynchronize(stream);

    // 8.结果打印
    for (int b = 0; b < ib; ++b) {
        printf("batch %d. output_data_host = \n", b);
        for (int i = 0; i < iw * ih; ++i) {
            printf("%f, ", output_data_host[b * iw * ih + i]);
            if ((i + 1) % iw == 0)
                printf("\n");
        }
    }

    printf("Clean memory\n");
    cudaStreamDestroy(stream);
    cudaFree(input_data_device);
    cudaFree(output_data_device);
    execution_context->destroy();
    engine->destroy();
    runtime->destroy();
    return;
}
