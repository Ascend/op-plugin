# torch_npu.optim.NpuFusedAdadelta

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

通过张量融合实现的高性能Adadelta优化器，核心功能和`torch.optim.Adadelta`兼容。

Adadelta的功能和原理可参考[ADADELTA](https://pytorch.org/docs/2.1/generated/torch.optim.Adadelta.html#adadelta)。

## 函数原型

```
class torch_npu.optim.NpuFusedAdadelta(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)
```


## 参数说明

- **params** (`dict`)：必选参数，模型参数或模型参数组，`params`为参数的可迭代对象或参数组的dict类型。
- **lr** (`float`)：可选参数，学习率，默认值为1e-3。`lr`的值小于0时，打印“ValueError”异常信息。
- **rho** (`float`)：可选参数，梯度的均方差系数，默认值为0.9。`rho`小于0或者`rho`大于1时，打印“ValueError”异常信息。
- **eps** (`float`)：可选参数，分母防止除0项，提高数值稳定性，默认值为1e-6。`eps`小于0时，打印“ValueError”异常信息。
- **weight_decay** (`float`)：可选参数，权重衰减，默认值为0。`weight_decay`小于0时，打印“ValueError”异常信息。


## 返回值说明

类型为`NpuFusedAdadelta`的对象。


## 约束说明

`NpuFusedAdadelta`的实现机制要求`params`中的每一个模型参数对象在使用过程中不能被重新申请，否则将导致无法预料的结果。引起模型参数对象被重新申请的操作包括但不限于：

- 将模型或其子Module进行.cpu操作
- 将模型参数对象指向新的对象
- 将模型参数对象置为None

对模型参数对象进行inplace计算，或者读取参数的值，`NpuFusedAdadelta`可正常工作。


## 调用示例

```python
import torch
from torch_npu.npu.amp import GradScaler, autocast
from torch_npu.optim import NpuFusedAdadelta 

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

opt_kwargs = dict(lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.05)
params = _create_simple_params_and_grads()
fused_opt = NpuFusedAdadelta(params, **opt_kwargs)
with torch.no_grad():
    fused_opt.step()
```

