
// #ifndef ONNX_PLUGIN_HPP
// #define ONNX_PLUGIN_HPP

#include <memory>
#include <vector>
#include <set>
#include <string>

#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>

namespace ONNXPlugin {

/*
 * 在没有显式指定的情况下，枚举值从 0 开始递增。
 * 因此 CompilePhase 的值是 0 InferencePhase 的值是 1
 */
enum Phase {
    CompilePhase,
    InferencePhase
};

typedef unsigned short float16;

/*
 * enum class 强类型枚举，相比于 enum 枚举 enum class 提供了更强的类型安全性。
 * DataType：这是枚举类型的名称。
 * ': int'：指定了底层类型为 int。这意味着每个枚举值实际上都是一个 int 类型的值。
 */
enum class DataType : int {
    Float32 = 0,
    Float16 = 1,
    Int32 = 2,
    UInt8 = 3
};

// 返回 DataType 所占字节数
inline int DataTypeSizeOf(DataType dt) {
    switch (dt) {
    case DataType::Float32: return 4;
    case DataType::Float16: return 2;
    case DataType::Int32: return 4;
    case DataType::UInt8: return 1;
    default: return 0;
    }
}

// DataType 转字符串
inline const char *data_type_string(DataType dt) {
    switch (dt) {
    case DataType::Float32: return "Float32";
    case DataType::Float16: return "Float16";
    case DataType::Int32: return "Int32";
    case DataType::UInt8: return "UInt8";
    default: return "UnknowDataType";
    }
}

struct GTensor {
    GTensor() {
    }
    GTensor(float *ptr, int ndims, int *dims);
    GTensor(float16 *ptr, int ndims, int *dims);

    /*
     * const 修饰变量，说明该变量不可以被改变；
     * const 修饰指针，分为指向常量的指针（pointer to const）和自身是常量的指针（常量指针，const pointer）；
     * const 修饰引用，指向常量的引用（reference to const），用于形参类型，即避免了拷贝，又避免了函数对值的修改；
     * const 修饰成员函数，说明该成员函数内不能修改成员变量。
     */
    int count(int start_axis = 0) const;

    /*
     * template <typename... _Args> 声明一个可变参数模板;
     * int index 是函数的第一个参数，表示第一个索引。
     * _Args &&...index_args 是函数的可变参数，表示剩余的索引参数。
     * index 是第一个索引，index_args... 是剩余的索引参数，展开成多个索引。
     * sizeof...(index_args) 计算可变参数的数量。
     */
    template <typename... _Args>
    int offset(int index, _Args &&...index_args) const {
        const int index_array[] = {index, index_args...};
        return offset_array(sizeof...(index_args) + 1, index_array);
    }

    int offset_array(const std::vector<int> &index) const;
    int offset_array(size_t size, const int *index_array) const;

    /*
     * 普通成员函数：普通成员函数通常在类定义中声明，但实现（定义）在类定义外部。
     * 内联成员函数：内联成员函数在类定义中既声明又定义。
     * pytorch的默认格式是[b,h,w,c]; tensorflow的默认格式是[b,c,h,w]
     */
    inline int batch() const {
        return shape_[0];
    }
    inline int channel() const {
        return shape_[1];
    }
    inline int height() const {
        return shape_[2];
    }
    inline int width() const {
        return shape_[3];
    }

    template <typename _T>
    inline _T *ptr() const {
        // (_T*) 也是类型转换，将变量强转为 _T* 类型
        return (_T *)ptr_;
    }

    template <typename _T, typename... _Args>
    inline _T *ptr(int i, _Args &&...args) const {
        return (_T *)ptr_ + offset(i, args...);
    }

    void *ptr_ = nullptr;
    DataType dtype_ = DataType::Float32;
    std::vector<int> shape_;
};

struct Weight {
    Weight() = default; // 告诉编译器生成一个默认构造函数；
    Weight(const std::vector<int> &dims, DataType dt);
    void to_float32();
    void to_float16();
    void copy_to_gpu();
    void free_host();
    void free();

    void *pdata_host_ = nullptr;
    void *pdata_device_ = nullptr;
    size_t data_bytes_ = 0;
    size_t numel_ = 0;
    std::vector<int> dims_;
    DataType dt_ = DataType::Float32;
    char shape_string_[100] = {0};
};

// 从数据流中读取数据
class InStream {
public:
    // 构造函数，接受一个指向数据的指针 pdata 和数据大小 size。
    InStream(const void *pdata, size_t size);

    // 成员函数，重载了 ">>" 运算符用于读取 std::string 类型的数据，返回一个 InStream 的引用
    InStream &operator>>(std::string &value);

    template <typename _T>
    InStream &operator>>(std::vector<_T> &value) {
        int size = 0;
        (*this) >> size; // 执行的是模板函数里的这行 read(&value, sizeof(_T)); 代码
        value.resize(size);
        for (int i = 0; i < size; ++i) {
            (*this) >> value[i];
        }
        return *this;
    }

    template <typename _T>
    InStream &operator>>(_T &value) {
        read(&value, sizeof(_T));
        return *this;
    }

    void read(void *pdata, size_t size);
    int cursor() {
        return cursor_;
    }

private:
    const unsigned char *pdata_ = nullptr;
    size_t size_ = 0;
    int cursor_ = 0;
};

// 将不同类型的数据写入到内部缓冲区 data_ 中。
class OutStream {
public:
    OutStream &operator<<(const char *value);
    OutStream &operator<<(const std::string &value);

    template <typename _T>
    OutStream &operator<<(const std::vector<_T> &value) {
        int size = value.size();
        (*this) << size;
        for (int i = 0; i < size; ++i)
            (*this) << value[i];
        return *this;
    }

    template <typename _T>
    OutStream &operator<<(const _T &value) {
        write(&value, sizeof(_T));
        return *this;
    }

    void write(const void *pdata, size_t size);
    // 返回的是一个指向 常量数组的引用
    const std::vector<unsigned char> &data() {
        return data_;
    }

private:
    std::vector<unsigned char> data_;
};

struct LayerConfig {
    int num_output_ = 1;                                         // 输出数量
    int num_input_ = 1;                                          // 输入数量
    size_t workspace_size_ = 0;                                  // 工作空间大小，计算时需要的临时内存的大小。有的层在计算时需要额外的内存来存储中间结果。
    int max_batch_size_ = 0;                                     // 最大 batch size
    std::set<nvinfer1::DataType> support_dtype_set_;             // 支持的数据类型集合
    std::set<nvinfer1::PluginFormat> support_plugin_format_set_; // 支持的插件格式集合

    std::vector<std::shared_ptr<Weight>> weights_; // 权重向量，存储层的权重参数。
    DataType usage_dtype_;                         // 使用的数据类型，表示当前层正在使用的数据类型。
    nvinfer1::PluginFormat usage_plugin_format_;   // 使用的插件格式，表示当前层正在使用的数据格式。
    std::string info_;                             // 信息字符串，可能存储层的一些描述信息或元数据。

    std::vector<unsigned char> serialize_data_; // 序列化数据向量，存储该层的序列化数据。用于保存和加载层的配置和参数。

    LayerConfig();
    void serialize_data_copy_to(void *buffer);
    int serialize();
    void deserialize(const void *ptr, size_t length);
    // 设置层的配置，包括信息字符串和权重参数。
    void setup(const std::string &info, const std::vector<std::shared_ptr<Weight>> &weights);
    // 用于将层的配置和参数序列化到输出流。
    virtual void seril(OutStream &out) {
    }
    // 用于从输入流中反序列化层的配置和参数。
    virtual void deseril(InStream &in) {
    }
    // 用于初始化层的配置和参数。
    virtual void init() {
    }
};

/*
 * class MyPlugin : public nvinfer1::IPluginV2DynamicExt {
 * public:
 *     SetupPlugin(MyPlugin)
 * };
 * 宏 SetupPlugin(MyPlugin) 将被展开为：
 * class MyPlugin : public nvinfer1::IPluginV2DynamicExt {
 * public:
 *     virtual const char* getPluginType() const noexcept override { return "MyPlugin"; }
 *     virtual const char* getPluginVersion() const noexcept override { return "1"; }
 *     virtual nvinfer1::IPluginV2DynamicExt* clone() const noexcept override { return new MyPlugin(*this); }
 * };
 *'#class_' 是预处理器中的字符串化操作符，它将宏参数 class_ 转换为对应的字符串字面量。
 */
#define SetupPlugin(class_)                                                  \
    virtual const char *getPluginType() const noexcept override {            \
        return #class_;                                                      \
    };                                                                       \
    virtual const char *getPluginVersion() const noexcept override {         \
        return "1";                                                          \
    };                                                                       \
    virtual nvinfer1::IPluginV2DynamicExt *clone() const noexcept override { \
        return new class_(*this);                                            \
    }

/* 'class_##PluginCreator__' 中 'class_' 是宏参数，如果传入的参数(class_)为 MySigmoid，则 class_##PluginCreator__ == MyPluginPluginCreator__*/
#define RegisterPlugin(class_)                                                                                                              \
    class class_##PluginCreator__ : public nvinfer1::IPluginCreator {                                                                       \
    public:                                                                                                                                 \
        const char *getPluginName() const noexcept override {                                                                               \
            return #class_;                                                                                                                 \
        }                                                                                                                                   \
        const char *getPluginVersion() const noexcept override {                                                                            \
            return "1";                                                                                                                     \
        }                                                                                                                                   \
        const nvinfer1::PluginFieldCollection *getFieldNames() noexcept override {                                                          \
            return &mFieldCollection;                                                                                                       \
        }                                                                                                                                   \
                                                                                                                                            \
        nvinfer1::IPluginV2DynamicExt *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept override {        \
            auto plugin = new class_();                                                                                                     \
            mFieldCollection = *fc;                                                                                                         \
            mPluginName = name;                                                                                                             \
            return plugin;                                                                                                                  \
        }                                                                                                                                   \
                                                                                                                                            \
        nvinfer1::IPluginV2DynamicExt *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override { \
            auto plugin = new class_();                                                                                                     \
            plugin->pluginInit(name, serialData, serialLength);                                                                             \
            mPluginName = name;                                                                                                             \
            return plugin;                                                                                                                  \
        }                                                                                                                                   \
                                                                                                                                            \
        void setPluginNamespace(const char *libNamespace) noexcept override {                                                               \
            mNamespace = libNamespace;                                                                                                      \
        }                                                                                                                                   \
        const char *getPluginNamespace() const noexcept override {                                                                          \
            return mNamespace.c_str();                                                                                                      \
        }                                                                                                                                   \
                                                                                                                                            \
    private:                                                                                                                                \
        std::string mNamespace;                                                                                                             \
        std::string mPluginName;                                                                                                            \
        nvinfer1::PluginFieldCollection mFieldCollection{0, nullptr};                                                                       \
    };                                                                                                                                      \
    REGISTER_TENSORRT_PLUGIN(class_##PluginCreator__);

class TRTPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    virtual nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept override {
        return inputTypes[0];
    }

    virtual void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override;

    // /*cublas*/ 这些注释表明这些参数在函数体内没有使用，通常用于避免编译器警告。
    virtual void attachToContext(cudnnContext * /*cudnn*/, cublasContext * /*cublas*/, nvinfer1::IGpuAllocator * /*allocator*/) noexcept override {
        // 从上下文分离，通常用于释放资源（在这里没有实际实现）
    }
    virtual void detachFromContext() noexcept override {
        // 从上下文分离，通常用于释放资源（在这里没有实际实现）
    }
    virtual void setPluginNamespace(const char *pluginNamespace) noexcept override {
        this->namespace_ = pluginNamespace;
    };
    virtual const char *getPluginNamespace() const noexcept override {
        return this->namespace_.data();
    };

    virtual ~TRTPlugin();
    virtual int enqueue(const std::vector<GTensor> &inputs, std::vector<GTensor> &outputs, const std::vector<GTensor> &weights, void *workspace, cudaStream_t stream) = 0;

    void pluginInit(const std::string &name, const std::string &info, const std::vector<std::shared_ptr<Weight>> &weights);
    void pluginInit(const std::string &name, const void *serialData, size_t serialLength);
    // 插件配置完成后的钩子函数
    virtual void config_finish(){};

    virtual std::shared_ptr<LayerConfig> new_config();
    virtual bool supportsFormatCombination(
        int32_t pos, const nvinfer1::PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    virtual int getNbOutputs() const noexcept;
    virtual nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs, nvinfer1::IExprBuilder &exprBuilder) noexcept {
        return inputs[0];
    }

    // 插件初始化函数，通常用于分配资源
    virtual int initialize() noexcept;
    // 插件终止函数，通常用于释放资源
    virtual void terminate() noexcept;
    virtual void destroy() noexcept override {
    }
    virtual size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc *outputs,
                                    int32_t nbOutputs) const noexcept override;

    virtual int32_t enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                            const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

    // 获取序列化大小
    virtual size_t getSerializationSize() const noexcept override;
    // 序列化插件，通常用于保存插件状态
    virtual void serialize(void *buffer) const noexcept override;

protected:
    std::string namespace_;               // 插件命名空间
    std::string layerName_;               // 层名称
    Phase phase_ = CompilePhase;          // 插件当前阶段
    std::shared_ptr<LayerConfig> config_; // 插件配置
    std::vector<GTensor> inputTensors_;   // 输入张量
    std::vector<GTensor> outputTensors_;  // 输出张量
    std::vector<GTensor> weightTensors_;  // 权重张量
};
}; // namespace ONNXPlugin

// #endif // ONNX_PLUGIN_HPP