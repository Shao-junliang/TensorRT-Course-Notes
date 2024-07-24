import onnx
import onnx.helper as helper
import numpy as np

model = onnx.load("./src/cuda-tensorrt-basic-api/static/mydemo.onnx")

# 打印模型
print("====================node信息====================")
print(model)

# 获取权重
conv_weight = model.graph.initializer[0]
conv_bias = model.graph.initializer[1]

# 数据是以protobuf的格式存储的，因此当中的数值会以bytes的类型保存，通过np.frombuffer方法还原成类型为float32的ndarray
print(f"===================={conv_weight.name}==========================")
print(conv_weight.name, np.frombuffer(conv_weight.raw_data, dtype=np.float32))

print(f"===================={conv_bias.name}==========================")
print(conv_bias.name, np.frombuffer(conv_bias.raw_data, dtype=np.float32))