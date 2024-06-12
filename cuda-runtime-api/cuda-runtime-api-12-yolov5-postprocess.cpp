#include "cuda-runtime-api.h"

std::vector<uint8_t> load_file(const std::string &file) {
    /*
     * ifstream：是C++标准库中的输入文件流类，用于从文件读取数据。
     * ios::in 表示以读模式打开文件，ios::binary 表示以二进制模式打开文件。
     */
    std::ifstream in(file, std::ios::in | std::ios::binary);

    // 检查文件是否成功打开。
    if (!in.is_open()) { return {}; }

    // 将文件读指针移动到文件的末尾。这是为了确定文件的长度。
    in.seekg(0, std::ios::end);

    // ：获取当前读指针的位置，由于之前调用了 seekg 将指针移动到文件末尾，因此此处 tellg() 返回文件的总长度。
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) {
        // 将文件读指针移动回文件的开始位置，准备读取文件内容。
        in.seekg(0, std::ios::beg);

        data.resize(length);

        // 从文件中读取 length 字节的数据，存储到 data 向量中。
        // 从 &data[0] 开始，这是向量中第一个元素的地址，将其强制转换为 char* 类型，以便与 read 函数的参数类型匹配。
        in.read((char *)&data[0], length);
    }
    in.close();
    return data;
}

std::vector<Box> cpu_decode(float *predict, int rows, int cols, float confidence_threshold, float nms_threshold) {
    std::vector<Box> boxes;
    int num_classes = cols - 5; // num_classes 是类别概率，一个目标框可能有多个类别概率，只是概率值不同，有高有低；
    // yolo推理结果存储格式：[[left, top, width, height, confidence,label1, label2,……],[……],……]
    for (int i = 0; i < rows; ++i) {
        // pitem 是指向每一行结果的首地址
        float *pitem = predict + i * cols;

        // 置信度，表示该预测框中是否有对象的置信度。
        float objness = pitem[4];

        // 第一次过滤，当置信度小于某个值时，直接过滤，免得后面的计算占时间；
        if (objness < confidence_threshold) { continue; }

        // 当前行的 label 指针，指向类别概率的起始位置。
        float *pclass = pitem + 5;

        // max_element() 找到类别概率数组中最大元素的指针。- pclass：计算最大元素的索引，即类别标签。
        int label = std::max_element(pclass, pclass + num_classes) - pclass;

        // 获取预测结果中指定类别的概率值。
        float prob = pclass[label]; //  pitem + 5[]

        // 概率值乘以置信度为最终的置信度分数；
        float confidence = prob * objness;

        // 第二次过滤
        if (confidence < confidence_threshold) { continue; }

        // 获取 left, top, width, height 的值
        float cx = pitem[0];
        float cy = pitem[1];
        float width = pitem[2];
        float height = pitem[3];

        // xywh to xyxy
        float left = cx - width * 0.5;
        float top = cy - height * 0.5;
        float right = cx + width * 0.5;
        float bottom = cy + height * 0.5;
        boxes.emplace_back(left, top, right, bottom, confidence, (float)label);
    }

    // 按照置信度排序，sort传参传引用可以避免一次拷贝；
    std::sort(boxes.begin(), boxes.end(), [](Box &a, Box &b) { return a.confidence > b.confidence; });
    // nms 过程中，是否删除某个框的标记信息；
    std::vector<bool> remove_flags(boxes.size());
    std::vector<Box> box_result;
    box_result.resize(boxes.size());
    auto iou = [](const Box &a, const Box &b) {
        float cross_left = std::max(a.left, b.left);
        float cross_top = std::max(a.top, b.top);
        float cross_right = std::min(a.right, b.right);
        float cross_bottom = std::min(a.bottom, b.bottom);
        // 计算交集
        float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
        // 计算并集
        float union_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top)
                           + std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top) - cross_area;
        if (cross_area == 0 || union_area == 0) { return 0.0f; }
        return cross_area / union_area;
    };

    // nms去重
    for (int i = 0; i < boxes.size(); ++i) {
        if (remove_flags[i]) { continue; }

        auto &ibox = boxes[i];
        box_result.emplace_back(ibox);
        for (int j = i + 1; j < boxes.size(); ++j) {
            if (remove_flags[j]) { continue; }

            auto &jbox = boxes[j];
            if (ibox.label == jbox.label) {
                if (iou(ibox, jbox) >= nms_threshold) { remove_flags[j] = true; }
            }
        }
    }

    return box_result;
}

std::vector<Box> gpu_decode(float *predict, int rows, int cols, float confidence_threshold, float nms_threshold) {
    std::vector<Box> box_result;
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    float *predict_device = nullptr;
    float *output_device = nullptr;
    float *output_host = nullptr;

    /*
     * 由于nms之后输出的结果长度是不确定对策，为了在gpu上输出数量不确定的数组，
     * 用[count, box1, box2, ……] 的方式表示，其中 count 是数组的数量，
     */
    int max_objects = 1000;
    // left, top, right, bottom, confidence, class, keepflag
    int NUM_BOX_ELEMENT = 7;
    checkRuntime(cudaMalloc(&predict_device, rows * cols * sizeof(float)));
    // 前面的 'sizeof(float) +' 表示一个浮点数的大小，是 count 的存储空间
    checkRuntime(cudaMalloc(&output_device, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
    checkRuntime(cudaMallocHost(&output_host, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));

    checkRuntime(cudaMemcpyAsync(predict_device, predict, rows * cols * sizeof(float), cudaMemcpyHostToDevice, stream));
    decode_kernel_invoker(predict_device, rows, cols - 5,
                          confidence_threshold, nms_threshold, nullptr,
                          output_device, max_objects, NUM_BOX_ELEMENT, stream);

    checkRuntime(cudaMemcpyAsync(output_host, output_device, sizeof(int) + max_objects * NUM_BOX_ELEMENT * sizeof(float), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    // 避免角标越界，因为 output_host 里面的 box 最大数量为 max_objects， 但是 count 的数量有可能超过 max_objects
    int num_boxes = std::min((int)output_host[0], max_objects);
    for (int i = 0; i < num_boxes; ++i) {
        float *ptr = output_host + 1 + NUM_BOX_ELEMENT * i;
        int keep_flag = ptr[6];
        if (!keep_flag) { continue; }
        box_result.emplace_back(ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], (int)ptr[5]);
    }

    // 释放内存
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFree(predict_device));
    checkRuntime(cudaFree(output_device));
    checkRuntime(cudaFreeHost(output_host));
    return box_result;
}

void cuda_runtime_api_12_yolov5_postprocess() {
    auto data = load_file("../src/cuda-runtime-api/static/predict.data");
    cv::Mat image = cv::imread("../src/cuda-runtime-api/static/12.input-image.jpg");
    float *ptr = (float *)data.data();
    int nelem = data.size() / sizeof(float);
    int ncols = 85;
    int nrows = nelem / ncols;
    auto boxes = cpu_decode(ptr, nrows, ncols);
    auto boxse = gpu_decode(ptr, nrows, ncols);
    for (auto &box : boxse) {
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2);
        cv::putText(image, cv::format("%.2f", box.confidence), cv::Point(box.left, box.top - 7), 0, 0.8, cv::Scalar(0, 0, 255), 2, 16);
    }

    cv::imwrite("../src/cuda-runtime-api/static/12.image-draw.jpg", image);
    return;
}