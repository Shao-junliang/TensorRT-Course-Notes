#include "cuda-tensorrt-api.h"

bool build_dynamic_shape_model() {
    TRTLogger logger;
    // --------------------------------- 1. 定义 builder、config、network 指针 ----------------------------------
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);

    /* --------------------------------- 2. 模型结构与输入输出信息 ----------------------------------
     * 构建的模型结构：
     * Network definition:
     * image
     *   |
     * conv(3x3, pad=1)  input = 1, output = 1, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5], [0.2, 0.2, 0.1]], b=0.0
     *   |
     * relu
     *   |
     * prob
     */
    // 输入输出的数量与网络权重的初始化
    const int num_input = 1;
    const int num_output = 1;
    float layer1_weight_values[] = {1.0, 2.0, 3.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2};
    float layer1_bias_values[] = {0.0};

    // 网络添加输入节点
    nvinfer1::ITensor *input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(-1, num_input, -1, -1));
    // 网络隐藏层权重初始化
    nvinfer1::Weights layer1_weight = CTA::make_weights(layer1_weight_values, 9);
    nvinfer1::Weights layer1_bias = CTA::make_weights(layer1_bias_values, 1);

    // 网络添加隐藏层节点
    auto layer1 = network->addConvolution(*input, num_output, nvinfer1::DimsHW(3, 3), layer1_weight, layer1_bias);
    // 隐藏层添加 padding 操作
    layer1->setPadding(nvinfer1::DimsHW(1, 1));
    // 隐藏层添加激活函数
    auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kRELU);
    // 将隐藏层的输出标记为模型输出
    network->markOutput(*prob->getOutput(0));

    // 超参初始化
    int maxBatchSize = 10;
    // 配置暂存存储器，用于layer实现的临时存储，也用于保存中间激活值
    config->setMaxWorkspaceSize(1 << 28);

    /* --------------------------------- 2.1 关于profile ----------------------------------
     * 如果模型有多个输入，则必须多个profile
     * 优化配置文件（Optimization Profile）允许你为动态输入张量设置最小（kMIN）、最优（kOPT）和最大（kMAX）维度。
     */
    auto profile = builder->createOptimizationProfile();
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
    FILE *f = fopen("../src/cuda-tensorrt-basic-api/static/dynamic_engine.trtmodel", "wb");
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

void cuda_tensorrt_basic_api_3_dynamic_shape() {
    if (!build_dynamic_shape_model()) {
        return;
    }
    TRTLogger logger;
    // 1.加载模型
    auto engine_data = CTA::load_file("../src/cuda-tensorrt-basic-api/static/dynamic_engine.trtmodel");
    // 2.创建runtime实例
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    // 3.模型反序列化
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (engine == nullptr) {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }
    // 4.创建上下文
    nvinfer1::IExecutionContext *execution_context = engine->createExecutionContext();
    // 5.创建CUDA流
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // print(F.conv2d(input, weight, bias, padding=1)) 网络结构

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

    // 6.2 输出y定义
    const int ib = 2;
    // 3x3输入，对应3x3输出
    const int iw = 3;
    const int ih = 3;
    float output_data_host[ib * iw * ih];
    float *output_data_device = nullptr;
    // 6.3 输入数据 cpu to gpu
    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));
    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);

    // 6.4 因为是动态输入，所以在模型推理之前，需明确输入数据的大小
    execution_context->setBindingDimensions(0, nvinfer1::Dims4(ib, 1, ih, iw));
    // 6.5 构建 bindings 数组
    float *bindings[] = {input_data_device, output_data_device};
    // 6.6 模型推理
    bool success = execution_context->enqueueV2((void **)bindings, stream, nullptr);
    // 6.7 模型输出结果 gpu to cpu
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    // 7.流同步
    cudaStreamSynchronize(stream);

    // 结果打印
    for (int b = 0; b < ib; ++b) {
        printf("batch %d. output_data_host = \n", b);
        for (int i = 0; i < iw * ih; ++i) {
            printf("%f, ", output_data_host[b * iw * ih + i]);
            if ((i + 1) % iw == 0) { printf("\n"); }
        }
    }
    printf("Clean memory\n");
    // 8.内存释放
    cudaStreamDestroy(stream);
    cudaFree(input_data_device);
    cudaFree(output_data_device);
    execution_context->destroy();
    engine->destroy();
    runtime->destroy();
    return;
}