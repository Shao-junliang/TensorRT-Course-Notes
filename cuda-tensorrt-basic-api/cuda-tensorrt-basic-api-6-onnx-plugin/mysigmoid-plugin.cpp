#include "mysigmoid-plugin.hpp"
#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"

#include <cassert>
#include <cstring>
#include <string.h>
#include <string>
#include <vcruntime.h>
#include <vcruntime_string.h>
#include <vector>

using namespace nvinfer1;
// MySigmoid plugin的特定常量
namespace {
const char *MYSIGMOID_PLUGIN_VERSION{"1"}; // 采用的名称要对应上onnx-tensorrt-release-8.0/builtin_op_importers.cpp:5094行定义的名称
const char *MYSIGMOID_PLUGIN_NAME{"MySigmoid"};
} // namespace

// 静态类字段的初始化
PluginFieldCollection MySigmoidPluginCreator::m_FC{}; // FieldCollection 字段收集
std::vector<PluginField> MySigmoidPluginCreator::m_PluginAttributes;

// 实际注册时，注册的是创建器，交给tensorRT管理
REGISTER_TENSORRT_PLUGIN(MySigmoidPluginCreator);

// 用于反序列化插件的Helper function
template <typename T>
T readFromBuffer(const char *&buffer) {
    T val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
    return val;
}

// 用于序列化插件的Helper function
template <typename T>
void writeToBuffer(char *&buffer, const T &val) {
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
}

// ======================================== 1.插件的具体实现 ========================================
// 定义插件类，插件类的构造函数
MySigmoidPlugin::MySigmoidPlugin(const std::string name, const std::string attr1, float attr3) :
    m_LayerName(name), m_attr1(attr1), m_attr3(attr3) {
    printf("==================== 编译阶段，attr1 = %s, attr3 = %f\n", attr1.c_str(), attr3);
}

MySigmoidPlugin::MySigmoidPlugin(const std::string name, const void *data, size_t length) {
    // 将 data 转换成 const char * 类型，以便读取字节数据。d 用于读取数据，a 用于记录数据的起始位置以便后续校验。
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    // 使用 readFromBuffer<int>(d) 读取 int 类型的数据，得到 nstr，nstr 为 int 类型数据的长度
    // readFromBuffer 负责从 d 指针所指向的内存位置读取一个 T 类型的值，并将 d 指针向前移动相应的字节数。
    int nstr = readFromBuffer<int>(d);
    m_attr1 = std::string(d, d + nstr);

    d += nstr; // 这一步是跳过 attr2 属性的内存位置
    m_attr3 = readFromBuffer<float>(d);
    assert(d == (a + length));
    printf("==================== 推理阶段，attr1 = %s, attr3 = %f\n", m_attr1.c_str(), m_attr3);
}

const char *MySigmoidPlugin::getPluginType() const noexcept {
    return MYSIGMOID_PLUGIN_NAME;
}

const char *MySigmoidPlugin::getPluginVersion() const noexcept {
    return MYSIGMOID_PLUGIN_VERSION;
}

int MySigmoidPlugin::getNbOutputs() const noexcept {
    return 1;
}

nvinfer1::DimsExprs MySigmoidPlugin::getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs, nvinfer1::IExprBuilder &exprBuilder) noexcept {
    // MySigmoidPlugin 不改变输入尺寸，所以输出尺寸将与输入尺寸相同
    return *inputs;
}

int MySigmoidPlugin::initialize() noexcept {
    return 0;
}

int MySigmoidPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                             const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
    void *output = outputs[0];
    size_t volume = 1;
    for (int i = 0; i < inputDesc->dims.nbDims; i++) {
        volume *= inputDesc->dims.d[i];
    }
    m_InputVolume = volume;
    mysigmoid_inference(
        static_cast<const float *>(inputs[0]),
        static_cast<float *>(output),
        m_InputVolume,
        stream);
    return 0;
}

size_t MySigmoidPlugin::getSerializationSize() const noexcept {
    return sizeof(int) + m_attr1.size() + sizeof(m_attr3);
}

// 自定义op层的参数序列化储存为trtmodel文件
void MySigmoidPlugin::serialize(void *buffer) const noexcept {
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    int nstr = m_attr1.size();
    writeToBuffer(d, nstr);
    memcpy(d, m_attr1.data(), nstr);

    d += nstr;
    writeToBuffer(d, m_attr3);
    assert(d == a + getSerializationSize());
}

// 判断该插件所支持的数据格式和类型
bool MySigmoidPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    auto type = inOut[pos].type;
    auto format = inOut[pos].format;
    // 这个插件只支持普通的浮点数，以及NCHW输入格式
    if (type == DataType::kFLOAT && format == PluginFormat::kLINEAR) {
        return true;
    } else {
        return false;
    }
}

void MySigmoidPlugin::terminate() noexcept {
}

void MySigmoidPlugin::destroy() noexcept {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

// 配置插件格式:告诉你目前这个层所采用的数据格式和类型
void MySigmoidPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs,
                                      const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept {
    auto type = in->desc.type;
    auto format = in->desc.format;
    assert(nbOutputs == 1);
    assert(type == DataType::kFLOAT);
    assert(format == PluginFormat::kLINEAR);
}

// 克隆插件
IPluginV2DynamicExt *MySigmoidPlugin::clone() const noexcept {
    printf("===================克隆插件=================\n");
    auto plugin = new MySigmoidPlugin(m_LayerName, m_attr1, m_attr3);
    plugin->setPluginNamespace(m_Namespace.c_str());
    return plugin;
}

void MySigmoidPlugin::setPluginNamespace(const char *libNamespace) noexcept {
    m_Namespace = libNamespace;
}

const char *MySigmoidPlugin::getPluginNamespace() const noexcept {
    return m_Namespace.c_str();
}

// ======================================== 2.插件创建器 ========================================
MySigmoidPluginCreator::MySigmoidPluginCreator() {
    m_PluginAttributes.emplace_back(PluginField("attr1", nullptr, PluginFieldType::kCHAR, 0));
    m_PluginAttributes.emplace_back(PluginField("attr3", nullptr, PluginFieldType::kFLOAT32, 1));

    m_FC.nbFields = m_PluginAttributes.size();
    m_FC.fields = m_PluginAttributes.data();
}

const char *MySigmoidPluginCreator::getPluginName() const noexcept {
    return MYSIGMOID_PLUGIN_NAME;
}

const char *MySigmoidPluginCreator::getPluginVersion() const noexcept {
    return MYSIGMOID_PLUGIN_VERSION;
}

const PluginFieldCollection *MySigmoidPluginCreator::getFieldNames() noexcept {
    return &m_FC;
}

// 创建插件
IPluginV2 *MySigmoidPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept {
    std::string attr1;
    float attr3;
    const PluginField *fields = fc->fields;

    for (int i = 0; i < fc->nbFields; i++) {
        /*
         * strcmp 是一个用于比较两个C风格字符串的标准库函数，按字典序比较两个字符串并返回一个整数。
         * 返回值小于0：表示 str1 小于 str2。等于0：表示 str1 等于 str2。大于0：表示 str1 大于 str2。
         */
        if (strcmp(fields[i].name, "attr1") == 0) {
            assert(fields[i].type == PluginFieldType::kCHAR);
            auto cp = static_cast<const char *>(fields[i].data);
            // 从一个字符指针 cp 开始，取指定长度的字符来初始化字符串
            attr1 = std::string(cp, cp + fields[i].length);
        } else if (strcmp(fields[i].name, "attr3") == 0) {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            attr1 = *(static_cast<const float *>(fields[i].data));
        }
    }
    return new MySigmoidPlugin(name, attr1, attr3);
}

// 反序列化创建插件
IPluginV2 *MySigmoidPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept {
    return new MySigmoidPlugin(name, serialData, serialLength);
}

void MySigmoidPluginCreator::setPluginNamespace(const char *libNamespace) noexcept {
    m_Namespace = libNamespace;
}

const char *MySigmoidPluginCreator::getPluginNamespace() const noexcept {
    return m_Namespace.c_str();
}