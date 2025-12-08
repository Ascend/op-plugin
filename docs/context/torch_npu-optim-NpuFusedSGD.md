# torch_npu.optim.NpuFusedSGD

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

通过张量融合实现的高性能SGD优化器，核心功能和`torch.optim.SGD`兼容。

SGD的功能和原理可参考[SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#sgd)。

## 函数原型

```
class torch_npu.optim.NpuFusedSGD(params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

## 参数说明

- **params** (`dict`)：必选参数，模型参数或模型参数组，`params`为参数的可迭代对象或参数组的dict类型。
- **lr** (`float`)：必选参数，学习率。`lr`不配置或值小于0时，打印“ValueError”异常信息。
- **momentum** (`float`)：可选参数，动量系数，默认值为0。`momentum`小于0时，打印“ValueError”异常信息。
- **dampening** (`float`)：可选参数，动量的抑制系数，默认值为0。
- **weight_decay** (`float`)：可选参数，权重衰减，默认值为0。`weight_decay`小于0时，打印“ValueError”异常信息。
- **nesterov** (`bool`)：可选参数，是否使用Nesterov动量，默认值为False。`nesterov`为True，同时`momentum`小于0或者`dampening`不等于0，打印“TypeError”异常信息。


## 返回值说明

类型为`NpuFusedSGD`的对象。


## 约束说明

`NpuFusedSGD`的实现机制要求`params`中的每一个模型参数对象在使用过程中不能被重新申请，否则将导致无法预料的结果。引起模型参数对象被重新申请的操作包括但不限于：

- 将模型或其子Module进行.cpu()操作
- 将模型参数对象指向新的对象
- 将模型参数对象置为None

对模型参数对象进行inplace计算，或者读取参数的值，`NpuFusedSGD`可正常工作。


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

