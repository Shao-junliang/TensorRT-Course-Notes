
/*
 * onnx解析器的头文件
 * 1. 如果 NvOnnxParser.h 在 CUDA\v11.7\include 中，则为已编译好的so解析器
 * 2. 如果用源代码编译，则直接 include 源代码中的 NvOnnxParser.h
 */
// #include <NvOnnxParser.h>
#include "../../../3rd_third/onnx-tensorrt/NvOnnxParser.h"

#include "cuda-tensorrt-api.h"

void cuda_tensorrt_basic_api_5_onnx_parser() {
    TRTLogger logger;
    // --------------------------------- 1. 定义 builder、config、network 指针 ----------------------------------
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);

    // --------------------------------- 2. 模型结构与输入输出信息 ----------------------------------
    // 输入输出的数量与网络权重的初始化，通过onnxparser解析器将模型的权重填充到network中，类似addConv的方式添加进去
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile("../src/cuda-tensorrt-basic-api/static/demo.onnx", 1)) {
        printf("Failed to parser demo.onnx\n");

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return;
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
        return;
    }

    // --------------------------------- 3. 序列化+保存 ----------------------------------
    nvinfer1::IHostMemory *model_data = engine->serialize();
    FILE *f = fopen("../src/cuda-tensorrt-basic-api/static/dynamic_engine_through_onnx_parser.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);
    // 内存释放
    model_data->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done.\n");
    return;
}
