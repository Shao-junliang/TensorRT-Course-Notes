#include "cuda-tensorrt-api.h"

void cuda_tensorrt_basic_api_2_inference() {
    // ------------------------------ 1. 加载模型   ----------------------------
    TRTLogger logger;
    // 读取模型文件
    auto engine_data = CTA::load_file("../src/cuda-tensorrt-basic-api/static/engine.trtmodel");
    // 执行推理前，需要创建一个推理的runtime接口实例。与builer一样，runtime需要logger：
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    // 将模型读取到engine_data中，对其进行反序列化以获得engine
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (engine == nullptr) {
        printf("Deserialize cuda engine failed.\n");
        delete runtime;
        return;
    }
    // 创建推理执行 上下文
    nvinfer1::IExecutionContext *execution_context = engine->createExecutionContext();
    // 创建CUDA流，以确定这个batch的推理是独立的
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // ------------------------------ 2. 准备好要推理的数据并搬运到GPU   ----------------------------
    float input_data_host[] = {1, 2, 3};
    float *input_data_device = nullptr;

    float output_data_host[2];
    float *output_data_device = nullptr;
    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));
    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);
    /*
     * bindings为指针数组，表示input和output在gpu中的指针。
     * 比如input有a，output有b, c, d，那么bindings = [a, b, c, d]，bindings[0] = a，bindings[2] = c。
     * 可以通过 engine->getBindingDimensions(0) 获取指定binding索引（这里是索引 0）的张量维度信息。
     * binding索引是指模型输入和输出张量在bindings数组中的位置。
     * binding索引：
     * 在 TensorRT 中，模型的输入和输出被称为 binding。这些binding在引擎创建时被确定，并且每个binding都有一个唯一的索引。
     * 在本节代码中，bindings 数组保存了输入和输出在 GPU 上的指针。
     * bindings[0] 是输入张量的指针。
     * bindings[1] 是输出张量的指针。
     */
    float *bindings[] = {input_data_device, output_data_device};
    // 获取第0个binding的维度
    nvinfer1::Dims inputDims = engine->getBindingDimensions(0);
    std::cout << "Input dimensions: ";
    for (int i = 0; i < inputDims.nbDims; ++i) {
        std::cout << inputDims.d[i] << " ";
    }
    std::cout << std::endl;

    // ------------------------------ 3. 推理并将结果搬运回CPU   ----------------------------
    bool success = execution_context->enqueueV2((void **)bindings, stream, nullptr);
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    printf("output_data_host = %f, %f\n", output_data_host[0], output_data_host[1]);

    // ------------------------------ 4. 释放内存 ----------------------------
    printf("Clean memory\n");
    cudaStreamDestroy(stream);
    delete execution_context;
    delete engine;
    delete runtime;

    // ------------------------------ 5. 手动推理进行验证 ----------------------------
    const int num_input = 3;
    const int num_output = 2;
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};
    float layer1_bias_values[] = {0.3, 0.8};
    printf("Manually verify calculation results:\n");
    for (int io = 0; io < num_output; ++io) {
        float output_host = layer1_bias_values[io];
        for (int ii = 0; ii < num_input; ++ii) {
            output_host += layer1_weight_values[io * num_input + ii] * input_data_host[ii];
        }
        float prob = 1 / (1 + exp(-output_host));
        printf("output_prob[%d] = %f\n", io, prob);
    }
}