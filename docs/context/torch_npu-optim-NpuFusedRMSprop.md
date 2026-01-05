# torch_npu.optim.NpuFusedRMSprop

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 训练系列产品</term>                                       |    √     |


## 功能说明

通过张量融合实现的高性能RMSprop优化器，核心功能和`torch.optim.RMSprop`兼容。

RMSprop的功能和原理可参考[RMSprop](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#rmsprop)。

## 函数原型

```
class torch_npu.optim.NpuFusedRMSprop(params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False)
```


## 参数说明

- **params** (`iterable`)：必选参数，模型参数或模型参数组。
- **lr** (`float`)：可选参数，学习率，默认值为1e-2。`lr`的值小于0时，打印“ValueError”异常信息。
- **alpha** (`float`)：可选参数，平滑常量，默认值为0.99。`alpha`的值小于0时，打印“ValueError”异常信息。
- **eps** (`float`)：可选参数，分母防止除零项，提高数值稳定性，默认值为1e-8。`eps`的值小于0时，打印“ValueError”异常信息。
- **weight_decay** (`float`)：可选参数，权重衰减，默认值为0。`weight_decay`的值小于0时，打印“ValueError”异常信息。
- **momentum** (`float`)：可选参数，动量因子，默认值为0。`momentum`的值小于0时，打印“ValueError”异常信息。
- **centered** (`bool`)：可选参数，计算中心RMSProp，梯度将被方差的估计值归一化，默认值为False。


## 返回值说明

类型为`NpuFusedRMSprop`的对象。


## 约束说明

`NpuFusedRMSprop`的实现机制要求`params`中的每一个模型参数对象在使用过程中不能被重新申请，否则将导致无法预料的结果。引起模型参数对象被重新申请的操作包括但不限于：

- 将模型或其子Module进行.cpu操作
- 将模型参数对象指向新的对象
- 将模型参数对象置为None

对模型参数对象进行inplace计算，或读取参数的值时，`NpuFusedRMSprop`都可以正常工作。


## 调用示例

```python
import torch
from torch_npu.npu.amp import GradScaler, autocast
from torch_npu.optim import NpuFusedRMSprop 

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

opt_kwargs = dict(eps=0.001, lr=0.01, weight_decay=1e-5)
params = _create_simple_params_and_grads()
fused_opt = NpuFusedRMSprop(params, **opt_kwargs)
with torch.no_grad():
    fused_opt.step()
```

