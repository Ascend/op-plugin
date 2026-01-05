# torch_npu.optim.NpuFusedAdamW

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

通过张量融合实现的高性能AdamW优化器，核心功能和`torch.optim.AdamW`兼容。

AdamW的功能和原理可参考[AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#adamw)。

## 函数原型

```
class torch_npu.optim.NpuFusedAdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
```


## 参数说明

- **params** (`iterable`)：必选参数，模型参数或模型参数组。
- **lr** (`float`)：可选参数，学习率，默认值为1e-3。`lr`的值小于0时，系统会抛出“ValueError”异常信息。
- **betas** (`Tuple[float, float]`)：可选参数，用于计算梯度及其平方的运行平均值的系数，`betas`为包含两个值的tuple类型，默认值为（0.9，0.999）。`betas`的值小于0或者`betas`的值大于1时，系统会抛出“ValueError”异常信息。
- **eps** (`float`)：可选参数，防止除0，提高数值稳定性，默认值为1e-8。`eps`小于0时，系统会抛出“ValueError”异常信息。
- **weight_decay** (`float`)：可选参数，权重衰减，默认值为1e-2。`weight_decay`小于0时，系统会抛出“ValueError”异常信息。
- **amsgrad** (`bool`)：可选参数，是否使用参考《[On the Convergence of Adam and Beyond](https://arxiv.org/pdf/1904.09237)》的AMSGrad变种实现，默认值为False。


## 返回值说明

类型为`NpuFusedAdamW`的对象。


## 约束说明

`NpuFusedAdamW`的实现机制要求`params`中的每一个模型参数对象在使用过程中不能被重新申请，否则将导致无法预料的结果。引起模型参数对象被重新申请的操作包括但不限于：

- 将模型或其子Module进行.cpu操作
- 将模型参数对象指向新的对象
- 将模型参数对象置为None

对模型参数对象进行inplace计算，或者读取参数的值，`NpuFusedAdamW`可正常工作。


## 调用示例

```python
import torch
from torch_npu.npu.amp import GradScaler, autocast
from torch_npu.optim import NpuFusedAdamW 

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

opt_kwargs = dict(eps=1e-8, betas=(0.9, 0.999), lr=2e-3, weight_decay=0.05)
params = _create_simple_params_and_grads()
fused_opt = NpuFusedAdamW(params, **opt_kwargs)
with torch.no_grad():
    fused_opt.step()
```

