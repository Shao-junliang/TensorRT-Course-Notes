import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torch.autograd
import json
from torch.onnx import OperatorExportTypes

class MySigmoidImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        # 这样写会报错：IndexError: Argument passed to at() was not in the map.
        # return x * 1 / (1 + torch.exp(-x))   
        return torch.sigmoid(x)
    
    @staticmethod
    def symbolic(g, x, p):
        return g.op("Plugin", x, p,
                    g.op("Constant", value_t=torch.tensor([3, 2, 1], dtype=torch.float32)),
                    name_s="MySigmoid",
                    info_s = json.dumps(
                        dict(attr1_s="string_attr",
                        attr2_i=[1, 2, 3],
                        attr3_f=222.0), ensure_ascii=False
                    )
                )
    
    
class MySigmoid(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.param = nn.parameter.Parameter(torch.arange(n).float())

    def forward(self, x):
        
        """
        torch.autograd.Function，创建自定义 autograd.Function 的基类
        1.前向传递中使用自定义自动求导函数，直接调用类方法 apply 。不要直接调用 forward() 。
        """
        return MySigmoidImpl.apply(x, self.param)
    

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.mysigmoid = MySigmoid(3)
        self.conv.weight.data.fill_(1)
        self.conv.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        x = self.mysigmoid(x)
        return x
    
if __name__ == "__main__":

    model = Model().eval()
    input = torch.tensor([
        # batch 0
        [
            [1,   1,   1],
            [1,   1,   1],
            [1,   1,   1],
        ],
        # batch 1
        [
            [-1,   1,   1],
            [1,   0,   1],
            [1,   1,   -1]
        ]
    ], dtype=torch.float32).view(2, 1, 3, 3)

    output = model(input)
    print(f"inference output = \n{output}")

    dummy = torch.zeros(1, 1, 3, 3)
    torch.onnx.export(
        model,
        # 输入给model的参数，需要传递tuple，因此用括号
        (dummy,),   
        # 模型保存路径
        "./src/cuda-tensorrt-basic-api/static/integrate_plugin_demo.onnx",   
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
        },
        # 对于插件，需要禁用onnx检查，这个参数只在 torch < 1.10.0 版本存在
        # enable_onnx_checker=False
        operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH
    )

    print("Done.!")


