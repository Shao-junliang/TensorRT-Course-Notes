import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import onnx.helper as helper
import numpy as np
import onnx
import os


def create_onnx_through_model_export():
    # 1.创建模型
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv = nn.Conv2d(1,1,3,padding=1)
            self.relu = nn.ReLU()
            self.conv.weight.data.fill_(1)
            self.conv.bias.data.fill_(0)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            return x
        
    print("对应opset文件夹代码在这里：", os.path.dirname(torch.onnx.__file__))

    model = Model()
    dummy = torch.zeros(1,1,3,3)
    # 2.模型导出onnx
    torch.onnx.export(
        model,
        # 输入给model的参数，需要传递tuple，因此用括号
        (dummy,),   
        # 模型保存路径
        "./src/cuda-tensorrt-basic-api/static/demo.onnx",   
        # 是否打印详细信息
        verbose=False,
        # 输入节点与输出节点名称
        input_names=["image"],
        output_names=["output"],
        # opset版本，
        opset_version=11,
        # 表示他有batch、height、width3个维度是动态的，在onnx中给其赋值为-1
        # 通常，我们只设置batch为动态，其他的避免动态
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }
    )
    print("Done.!")

def create_onnx_through_onnx_node():
    nodes = [
        helper.make_node(
            name="Conv_0",
            op_type="Conv",
            # 各个输入的名字，结点的输入包含：输入和算子的权重。必有输入X和权重W，偏置B可以作为可选。
            inputs=["image", "conv.weight", "conv.bias"],
            outputs=["3"],
            pads=[1,1,1,1],
            group=1,
            dilations=[1,1],
            kernel_shape=[3,3],
            strides=[1,1]
        ),
        helper.make_node(
            name="ReLU_1",
            op_type="Relu",
            inputs=["3"],
            outputs=["output"]
        )
    ]

    initializer = [
        helper.make_tensor(
            name="conv.weight",
            data_type=helper.TensorProto.DataType.FLOAT,
            dims=[1, 1, 3, 3],
            vals=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32).tobytes(),
            # raw(bool): 如果为真，则 vals 包含张量的序列化内容，否则，vals 应该是 *data_type* 定义的类型的值列表
            raw=True
        ),
        helper.make_tensor(
            name="conv.bias",
            data_type=helper.TensorProto.DataType.FLOAT,
            dims=[1],
            vals=np.array([0.0],dtype=np.float32).tobytes(),
            raw=True
        )
    ]

    inputs = [
        helper.make_tensor_value_info(
            name="image",
            elem_type=helper.TensorProto.DataType.FLOAT,
            shape=["batch", 1, 3, 3]
        )
    ]

    outputs = [
        helper.make_tensor_value_info(
            name="output",
            elem_type=helper.TensorProto.DataType.FLOAT,
            shape=["batch", 1, 3, 3]
        )
    ]

    graph = helper.make_graph(
        name="mymodel",
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        initializer=initializer
    )

    opset = [helper.make_operatorsetid("ai.onnx", 11)]

    model = helper.make_model(graph, opset_imports=opset, producer_name="pytorch", producer_version="1.9")
    onnx.save(model,"./src/cuda-tensorrt-basic-api/static/mydemo.onnx")

    print(model)
    print("Done.!")

if __name__ == '__main__':
    create_onnx_through_model_export()
    create_onnx_through_onnx_node()