#ifndef CUSTOM_MYSIGMOID_PLUGIN_H
#define CUSTOM_MYSIGMOID_PLUGIN_H

#include <NvInferPlugin.h>
#include <string>
#include <vector>

using namespace nvinfer1;

void mysigmoid_inference(const float *x, float *output, int n, cudaStream_t stream);

class MySigmoidPlugin : public IPluginV2DynamicExt {
public:
    MySigmoidPlugin(const std::string name, const std::string attr1, float attr3);

    MySigmoidPlugin(const std::string name, const void *data, size_t length);

    // It doesn't make sense to make MySigmoidPlugin without arguments, so we delete default constructor.
    MySigmoidPlugin() = delete;

    // ============================== 1.IPluginV2DynamicExt 类的纯虚函数 ==============================

    // 输出数据的尺寸
    virtual DimsExprs getOutputDimensions(int32_t outputIndex, DimsExprs const *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override;

    // 支持的数据类型，int8，float16，float32等
    virtual bool supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut,
                                           int32_t nbInputs, int32_t nbOutputs) noexcept override;

    //  配置插件格式
    virtual void configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs,
                                 DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept;

    // 需要的额外空间大小
    virtual size_t getWorkspaceSize(PluginTensorDesc const *inputs, int32_t nbInputs, PluginTensorDesc const *outputs,
                                    int32_t nbOutputs) const noexcept override {
        return 0;
    }

    // 推理具体逻辑
    virtual int32_t enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc,
                            void const *const *inputs, void *const *outputs,
                            void *workspace, cudaStream_t stream) noexcept override;

    // ============================== 2.IPluginV2Ext 类的纯虚函数 ==============================
    virtual nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const *inputTypes, int32_t nbInputs) const noexcept override {
        return inputTypes[0];
    }

    // ============================== 3.IPluginV2 类的纯虚函数 ==============================
    const char *getPluginType() const noexcept override;

    const char *getPluginVersion() const noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getSerializationSize() const noexcept override;

    int getNbOutputs() const noexcept override;

    void serialize(void *buffer) const noexcept override;

    void destroy() noexcept override;

    nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;

    void setPluginNamespace(const char *pluginNamespace) noexcept override;

    const char *getPluginNamespace() const noexcept override;

private:
    const std::string m_LayerName;
    std::string m_attr1;
    float m_attr3;
    size_t m_InputVolume;
    std::string m_Namespace;
};

class MySigmoidPluginCreator : public IPluginCreator {
public:
    MySigmoidPluginCreator();

    /*
     * const char*：函数的返回类型是一个指向常量字符的指针。
     * const: 类型限定符，表示这个成员函数是 const 的。const 成员函数保证不会修改它所属对象的成员变量。
     * noexcept: 异常说明符，表示这个函数不会抛出异常。
     * override: 表示这个成员函数重载了基类中的虚函数。使用 override 可以让编译器检查该函数是否确实覆盖了基类中的某个虚拟函数，如果没有覆盖成功，编译器会报错。
     */
    const char *getPluginName() const noexcept override;

    const char *getPluginVersion() const noexcept override;

    const PluginFieldCollection *getFieldNames() noexcept override;

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override;

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char *pluginNamespace) noexcept override;

    const char *getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection m_FC;
    static std::vector<PluginField> m_PluginAttributes;
    std::string m_Namespace;
};

#endif