import onnx
import onnx.helper as helper
import numpy as np

model = onnx.load("./src/cuda-tensorrt-basic-api/static/demo.onnx")

# 获取权重
conv_weight = model.graph.initializer[0]
conv_bias = model.graph.initializer[1]

# 修改权重
conv_weight.raw_data = np.arange(9,dtype=np.float32).tobytes()
# 重新保存
onnx.save(model, "./src/cuda-tensorrt-basic-api/static/change_demo.onnx")
print("Done.!")