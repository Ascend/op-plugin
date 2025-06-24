# torch_npu.optim.NpuFusedRMSpropTF

## 函数原型

```
class torch_npu.optim.NpuFusedRMSpropTF(params, lr=1e-2, alpha=0.9, eps=1e-10, weight_decay=0, momentum=0., centered=False, decoupled_decay=False, lr_in_momentum=True)
```

## 功能说明

通过张量融合实现的高性能RMSpropTF优化器，核心功能和torch.optim.RMSprop兼容。RMSpropTF的功能和原理可参考[https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/rmsprop_tf.py\#L14](https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/rmsprop_tf.py#L14)。

## 参数说明

- params：模型参数或模型参数组。
- lr：学习率，float类型（默认值：1e-3）。
- alpha：平滑常量，float类型（默认值：0.9）。
- eps：分母防止除0项，提高数值稳定性，float类型（默认值：1e-10）。
- weight_decay：权重衰减，float类型（默认值：0）。
- momentum：动量因子，float类型（默认值：0）。
- centered：计算中心RMSProp，梯度将被方差的估计值归一化，bool值（默认值：False）。
- decoupled_decay：权重衰减仅作用于参数，bool值（默认值：False）。
- lr_in_momentum：计算动量buffer时使用lr，bool值（默认值：True）。

## 输入说明

params为参数的可迭代对象或参数组的dict类型。

## 输出说明

类型为“NpuFusedRMSpropTF”的对象。

## 异常说明

- “ValueError”- “lr”值小于0。
- “ValueError”- “eps”的值小于0。
- “ValueError”- “momentum”的值小于0。
- “ValueError”- “weight_decay”的值小于0。
- “ValueError”- “alpha”的值小于0。

## 约束说明

NpuFusedRMSpropTF的实现机制要求params中的每一个模型参数对象在使用过程中不能被重新申请，否则将导致无法预料的结果。引起模型参数对象被重新申请的操作包括但不限于：

- 将模型或其子Module进行.cpu操作
- 将模型参数对象指向新的对象
- 将模型参数对象置为None

对模型参数对象进行inplace计算，或者读取参数的值，NpuFusedRMSpropTF可正常工作。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

```python
import torch
from torch_npu.npu.amp import GradScaler, autocast
from torch_npu.optim import NpuFusedRMSpropTF 

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
fused_opt = NpuFusedRMSpropTF(params, **opt_kwargs)
with torch.no_grad():
    fused_opt.step()
```

