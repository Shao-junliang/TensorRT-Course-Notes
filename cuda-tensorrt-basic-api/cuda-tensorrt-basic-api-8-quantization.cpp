#include <NvOnnxParser.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <NvInferRuntimeCommon.h>
#include "cuda-tensorrt-api.h"
#include "../cuda-runtime-api/utils.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#ifdef _WIN32
#include <windows.h>
#include <Shlwapi.h>                // 包含 PathFileExistsA 所需的头文件
#pragma comment(lib, "Shlwapi.lib") // 链接 Shlwapi.lib 库
#else
#include <unistd.h>
#endif

/*
 * typedef定义了一个类型别名Int8Process，它是一个特定的std::function类型。
 * 具体来说，Int8Process是一个可调用对象（函数、lambda表达式、函数对象等）的类型，这种可调用对象接受一组参数并返回void。
 */
typedef std::function<void(
    int current, int count, const std::vector<std::string> &files,
    nvinfer1::Dims dims, float *ptensor)>
    Int8Process;

// int8熵校准器：用于评估量化前后的分布改变
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(const std::vector<std::string> &imagefiles, nvinfer1::Dims dims, const Int8Process &preprocess) {
        assert(preprocess != nullptr);
        this->dims_ = dims;
        this->allimgs_ = imagefiles;
        this->preprocess_ = preprocess;
        this->fromCalibratorData_ = false;
        files_.resize(dims.d[0]);
    }

    // 这个构造函数，是允许从缓存数据中加载标定结果，这样不用重新读取图像处理
    Int8EntropyCalibrator(const std::vector<std::uint8_t> &entropyCalibratorData,
                          nvinfer1::Dims dims, const Int8Process &preprocess) {
        assert(preprocess != nullptr);
        this->dims_ = dims;
        this->entropyCalibratorData_ = entropyCalibratorData;
        this->preprocess_ = preprocess;
        this->fromCalibratorData_ = true;
        files_.resize(dims.d[0]);
    }

    virtual ~Int8EntropyCalibrator() {
        if (tensor_host_ != nullptr) {
            checkRuntime(cudaFreeHost(tensor_host_));
            checkRuntime(cudaFree(tensor_device_));
            tensor_host_ = nullptr;
            tensor_device_ = nullptr;
        }
    }

    int getBatchSize() const noexcept {
        return dims_.d[0];
    }

    bool next() {
        int batch_size = dims_.d[0];
        if (cursor_ + batch_size > allimgs_.size()) { return false; }

        for (int i = 0; i < batch_size; ++i) { files_[i] = allimgs_[cursor_++]; }

        if (tensor_host_ == nullptr) {
            size_t volumn = 1;
            for (int i = 0; i < dims_.nbDims; ++i) { volumn *= dims_.d[i]; }

            bytes_ = volumn * sizeof(float);

            checkRuntime(cudaMallocHost(&tensor_host_, bytes_));
            checkRuntime(cudaMalloc(&tensor_device_, bytes_));
        }

        preprocess_(cursor_, allimgs_.size(), files_, dims_, tensor_host_);
        checkRuntime(cudaMemcpy(tensor_device_, tensor_host_, bytes_, cudaMemcpyHostToDevice));
        return true;
    }

    bool getBatch(void *bindings[], const char *names[], int nbBindings) noexcept {
        if (!next()) { return false; }
        bindings[0] = tensor_device_;
        return true;
    }

    const std::vector<uint8_t> &getEntropyCalibratorData() {
        return entropyCalibratorData_;
    }

    const void *readCalibrationCache(size_t &length) noexcept {
        if (fromCalibratorData_) {
            length = this->entropyCalibratorData_.size();
            return this->entropyCalibratorData_.data();
        }
        length = 0;
        return nullptr;
    }

    virtual void writeCalibrationCache(const void *cache, size_t length) noexcept {
        const uint8_t *cacheStart = reinterpret_cast<const uint8_t *>(cache);
        entropyCalibratorData_.assign(cacheStart, cacheStart + length);
    }

private:
    Int8Process preprocess_;
    std::vector<std::string> allimgs_;
    size_t batchCudaSize_ = 0;
    int cursor_ = 0;
    size_t bytes_ = 0;
    nvinfer1::Dims dims_;
    std::vector<std::string> files_;
    float *tensor_host_ = nullptr;
    float *tensor_device_ = nullptr;
    std::vector<std::uint8_t> entropyCalibratorData_;
    bool fromCalibratorData_ = false;
};

// 通过智能指针管理nv返回的指针参数，内存自动释放，避免泄漏
template <typename _T>
static std::shared_ptr<_T> make_nvshared(_T *ptr) {
    return std::shared_ptr<_T>(ptr, [](_T *p) { p->destroy(); });
}

static bool exists(const std::string &path) {
#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

bool build_int8_model() {
    if (exists("engine.trtmodel")) {
        printf("Engine.trtmodel has exists.\n");
        return true;
    }
    TRTLogger logger;

    // 这是基本需要的组件
    auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    auto config = make_nvshared(builder->createBuilderConfig());
    auto network = make_nvshared(builder->createNetworkV2(1));

    auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile("../src/cuda-tensorrt-basic-api/static/classifier.onnx", 1)) {
        printf("Failed to parse classifier.onnx\n");

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
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();

    input_dims.d[0] = 1;
    config->setFlag(nvinfer1::BuilderFlag::kINT8);

    auto preprocess = [](int current, int count, const std::vector<std::string> &files,
                         nvinfer1::Dims dims, float *ptensor) {
        printf("Preprocess %d / %d\n", count, current);

        // 标定所采用的数据预处理必须与推理时一样
        int width = dims.d[3];
        int height = dims.d[2];
        float mean[] = {0.406, 0.456, 0.485};
        float std[] = {0.225, 0.224, 0.229};

        for (int i = 0; i < files.size(); ++i) {
            auto image = cv::imread(files[i]);
            cv::resize(image, image, cv::Size(width, height));
            int image_area = width * height;
            unsigned char *pimage = image.data;
            float *phost_b = ptensor + image_area * 0;
            float *phost_g = ptensor + image_area * 1;
            float *phost_r = ptensor + image_area * 2;
            for (int i = 0; i < image_area; ++i, pimage += 3) {
                // rgb顺序调换了
                *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
                *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
                *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
            }
            ptensor += image_area * 3;
        }
    };

    // 配置int8标定数据读取工具
    std::shared_ptr<Int8EntropyCalibrator> calib(new Int8EntropyCalibrator(
        {"../src/cuda-tensorrt-basic-api/static/kej.jpg"}, input_dims, preprocess));
    config->setInt8Calibrator(calib.get());

    // 配置最小允许batch
    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);

    // 配置最大允许batch
    input_dims.d[0] = maxBatchSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);

    auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
    if (engine == nullptr) {
        printf("Build engine failed.\n");
        return false;
    }

    // 将模型序列化，并储存为文件
    auto model_data = make_nvshared(engine->serialize());
    FILE *f = fopen("../src/cuda-tensorrt-basic-api/static/classifier_int8.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    f = fopen("../src/cuda-tensorrt-basic-api/static/calib.txt", "wb");
    auto calib_data = calib->getEntropyCalibratorData();
    fwrite(calib_data.data(), 1, calib_data.size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    printf("Done.\n");
    return true;
}

std::vector<std::string> load_labels(const char *file) {
    std::vector<std::string> lines;

    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        printf("open %d failed.\n", file);
        return lines;
    }

    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    in.close();
    return lines;
}

void cuda_tensorrt_basic_api_8_quantization() {
    if (!build_int8_model()) {
        return;
    }

    TRTLogger logger;
    auto engine_data = CTA::load_file("../src/cuda-tensorrt-basic-api/static/classifier_int8.trtmodel");
    auto runtime = make_nvshared(nvinfer1::createInferRuntime(logger));
    auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (engine == nullptr) {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    auto execution_context = make_nvshared(engine->createExecutionContext());

    int input_batch = 1;
    int input_channel = 3;
    int input_height = 224;
    int input_width = 224;
    int input_numel = input_batch * input_channel * input_height * input_width;
    float *input_data_host = nullptr;
    float *input_data_device = nullptr;
    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

    auto image = cv::imread("../src/cuda-tensorrt-basic-api/static/kej.jpg");
    float mean[] = {0.406, 0.456, 0.485};
    float std[] = {0.225, 0.224, 0.229};

    cv::resize(image, image, cv::Size(input_width, input_height));
    int image_area = image.rows * image.cols;
    unsigned char *pimage = image.data;
    float *phost_b = input_data_host + image_area * 0;
    float *phost_g = input_data_host + image_area * 1;
    float *phost_r = input_data_host + image_area * 2;
    for (int i = 0; i < image_area; ++i, pimage += 3) {
        *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
        *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
        *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
    }

    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

    const int num_classes = 1000;
    float output_data_host[num_classes];
    float *output_data_device = nullptr;
    checkRuntime(cudaMalloc(&output_data_device, sizeof(output_data_host)));

    auto input_dims = execution_context->getBindingDimensions(0);
    input_dims.d[0] = input_batch;

    execution_context->setBindingDimensions(0, input_dims);
    float *bindings[] = {input_data_device, output_data_device};
    bool success = execution_context->enqueueV2((void **)bindings, stream, nullptr);
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    float *prob = output_data_host;
    int predict_label = std::max_element(prob, prob + num_classes) - prob;
    auto labels = load_labels("../src/cuda-tensorrt-basic-api/static/labels.imagenet.txt");
    auto predict_name = labels[predict_label];
    float confidence = prob[predict_label];
    printf("Predict: %s, confidence = %f, label = %d\n", predict_name.c_str(), confidence, predict_label);

    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));
}
