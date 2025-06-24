# torch_npu.npu.disable_deterministic_with_backward

## 功能说明

关闭“确定性”功能。确定性算法是指在模型的前向传播过程中，每次输入相同，输出也相同。确定性算法可以避免模型在每次前向传播时产生的小随机误差累积，在需要重复测试或比较模型性能时非常有用。

## 函数原型

```
torch_npu.npu.disable_deterministic_with_backward(tensor) -> Tensor
```

## 参数说明

**tensor** (`Tensor`)：该接口为透明传输接口，不做数据处理，类型支持和数据格式为PyTorch在各芯片上的可支持的数据类型和数据格式，无接口级别的约束。

## 返回值
`Tensor`

代表`disable_deterministic_with_backward`的计算结果。

## 约束说明

- 入参`tensor`需要是训练网络中可以传递下去和整网的`output`有关联的tensor变量，否则无法进行反向设置确定性能力。
- 不支持图模式。

## 支持的型号

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> 
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> 
- <term>Atlas 训练系列产品</term> 
- <term>Atlas 推理系列产品</term> 
- <term>Atlas 200I/500 A2 推理产品</term> 

## 调用示例

单算子模式调用：

```python
import unittest
import torch
import torch.nn as nn
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

class TorchairSt(unittest.TestCase):
    def test_enable_to_disable_deterministic_algorithms(self):
        target_dtype = torch.float16        
        class DeterministicModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = x + 1
                x = torch_npu.npu.enable_deterministic_with_backward(x)
                add4 = x + y
                add1 = sum(add4)
                add1 = torch_npu.npu.disable_deterministic_with_backward(add1)
                add6 = add1 + add1
                return add6

        device = torch.device("npu:0")
        model = DeterministicModel()
        npu_mode = model.to(device)
        
        ins1 = torch.ones((2, 2), requires_grad=True).to(target_dtype).npu()
        ins2 = torch.ones((2, 2), requires_grad=True).to(target_dtype).npu()
        output_data = npu_mode(ins1, ins2)
        self.assertEqual(False, torch.are_deterministic_algorithms_enabled())

        loss_fn = nn.MSELoss()
        target_data = torch.randn((1, 2), requires_grad=True).to(target_dtype).npu()
        loss = loss_fn(output_data, target_data)
        loss.backward()
        self.assertEqual(False, torch.are_deterministic_algorithms_enabled())
    
if __name__ == '__main__':    
    unittest.main()

# 执行上述代码的输出类似如下：
----------------------------------------------------------------------
Ran 1 test in 4.636s

OK
```

