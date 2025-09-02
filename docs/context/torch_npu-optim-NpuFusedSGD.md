# torch_npu.optim.NpuFusedSGD

## 函数原型

```
class torch_npu.optim.NpuFusedSGD(params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

## 功能说明

通过张量融合实现的高性能SGD优化器，核心功能和torch.optim.SGD兼容。

SGD的功能和原理可参考[https://pytorch.org/docs/2.1/generated/torch.optim.SGD.html\#sgd](https://pytorch.org/docs/2.1/generated/torch.optim.SGD.html#sgd)。

## 参数说明

- params：模型参数或模型参数组。
- lr：学习率，float类型。
- momentum：动量系数，float类型（默认值：0）。
- dampening：动量的抑制系数，float类型（默认值：0）。
- weight_decay：权重衰减，float类型（默认值：0）。
- nesterov：是否使用Nesterov动量，bool类型（默认值：False）。

## 输入说明

params为参数的可迭代对象或参数组的dict类型。

## 输出说明

类型为“NpuFusedSGD”的对象。

## 异常说明

- “ValueError”- “lr”不提供或值小于0。
- “ValueError”- “momentum”小于0。
- “ValueError”- “weight_decay”小于0。
- “TypeError ”- “nesterov”为True，同时momentum小于0或者dampening不等于0。

## 约束说明

NpuFusedSGD的实现机制要求params中的每一个模型参数对象在使用过程中不能被重新申请，否则将导致无法预料的结果。引起模型参数对象被重新申请的操作包括但不限于：

- 将模型或其子Module进行.cpu()操作
- 将模型参数对象指向新的对象
- 将模型参数对象置为None

对模型参数对象进行inplace计算，或者读取参数的值，NpuFusedSGD可正常工作。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

```python
import torch
from torch_npu.npu.amp import GradScaler, autocast
from torch_npu.optim import NpuFusedSGD

def _create_simple_params_and_grads():
    params = [
        torch.arange(6).reshape(2, 3).float().npu(),
        torch.arange(12).reshape(4, 3).float().npu(),
        torch.arange(6).reshape(2, 3).half().npu(),
        torch.arange(12).reshape(4, 3).half().npu(),
        torch.arange(15).reshape(5, 3).float().npu(),
        torch.arange(18).reshape(6, 3).half().npu(),
        torch.arange(6).reshape(2, 3).float().npu()
    ]

    for i, p in enumerate(params):
        if i < len(params) - 1:
            p.requires_grad = True
            p.grad = p.clone().detach() / 100.

    return params

opt_kwargs = dict(lr=0.01, momentum=0.9, weight_decay=0.001)
params = _create_simple_params_and_grads()
fused_opt = NpuFusedSGD(params, **opt_kwargs)
with torch.no_grad():
    fused_opt.step()
```

