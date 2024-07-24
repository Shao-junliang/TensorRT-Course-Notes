#include "cuda-tensorrt-api.h"

// 本节代码主要实现一个最简单的神经网络 figure/simple_fully_connected_network.jpg
void cuda_tensorrt_basic_api_1_builder() {
    TRTLogger logger; // 输出日志，用来捕捉warning与info等

    /* ----------------------------- 1. 定义 builder, config 和network -----------------------------
     * 这是基本需要的组件
     * 形象的理解是你需要一个builder去build这个网络，网络自身有结构，这个结构可以有不同的配置
     */
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    // 创建一个构建配置，指定TensorRT应该如何优化模型，tensorRT生成的模型只能在特定配置下运行
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    /*
     * 创建网络定义，其中 createNetworkV2(1) 表示采用显性batch size，
     * 新版tensorRT(>=7.0)时，不建议采用0非显性batch size
     * 因此贯穿以后，请都采用 createNetworkV2(1) 而非 createNetworkV2(0) 或者 createNetwork
     */
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);
    /*
     * 构建一个模型
     * Network definition:
     *    image(此处输入是一个RGB的像素点，也可以理解为一个1x1的rgb图像)
     *      |
     *    linear (fully connected)  input = 3, output = 2, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5]], b=[0.3, 0.8]
     *      |
     *    sigmoid
     *      |
     *     prob
     */

    // ----------------------------- 2. 输入，模型结构和输出的基本信息 -----------------------------
    const int num_input = 3;  // 输入尺寸
    const int num_output = 2; // 输出尺寸
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};
    float layer1_bias_values[] = {0.3, 0.8};

    nvinfer1::ITensor *input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(1, num_input, 1, 1));
    nvinfer1::Weights layer1_weight = CTA::make_weights(layer1_weight_values, 6);
    nvinfer1::Weights layer1_bias = CTA::make_weights(layer1_bias_values, 2);
    // 添加全连接层, 注意：对input进行了解引用
    auto layer1 = network->addFullyConnected(*input, num_output, layer1_weight, layer1_bias);
    // 添加激活层, 注意：更严谨的写法是*(layer1->getOutput(0)) 即对getOutput返回的指针进行解引用
    auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    // 将需要的 prob 标记为模型输出
    network->markOutput(*prob->getOutput(0));
    // '1 << 28' 是 2的28次方 的意思
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f); // 256Mib
    config->setMaxWorkspaceSize(1 << 28);
    builder->setMaxBatchSize(1);

    // ----------------------------- 3. 生成engine模型文件 -----------------------------
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr) {
        printf("Build engine failed.\n");
        return;
    }

    /*
     * ----------------------------- 4. 序列化模型文件并存储 -----------------------------
     * engine 是一个指向已构建的 TensorRT 引擎的指针，类型为 nvinfer1::ICudaEngine*。
     * serialize() 是 ICudaEngine 类的一个成员函数。它将引擎对象序列化为一个可在磁盘上存储的字节流。
     * serialize() 返回一个指向 IHostMemory 的指针，IHostMemory 是一个接口，表示在主机内存中的一个连续的内存块。
     */
    nvinfer1::IHostMemory *model_data = engine->serialize();
    // "wb" 表示以二进制写入模式打开文件。如果文件不存在，将创建一个新文件。如果文件已存在，则会覆盖其内容。
    FILE *f = fopen("../src/cuda-tensorrt-basic-api/static/engine.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    model_data->destroy();
    delete engine;
    delete network;
    delete config;
    builder->destroy();
    printf("Done.\n");
    return;
}