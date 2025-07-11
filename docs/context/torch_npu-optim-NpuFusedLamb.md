# torch_npu.optim.NpuFusedLamb

## 函数原型

```
class torch_npu.optim.NpuFusedLamb(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, adam=False, use_global_grad_norm=False)
```

## 功能说明

通过张量融合实现的高性能Lamb优化器，原理可参考《[Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/pdf/1904.00962)》。

## 参数说明

- params：模型参数或模型参数组。
- lr：学习率，float类型（默认值：1e-3）。
- betas：用于计算梯度及其平方的运行平均值的系数，类型为Tuple[float, float]（默认值：（0.9，0.999））。
- eps：分母防止除0项，提高数值稳定性，float类型（默认值：1e-8）。
- weight_decay：权重衰减，float类型（默认值：0）。
- adam：是否通过将trust ratio设置为1，退化为Adam，bool类型（默认值：False）。
- use_global_grad_norm：是否使用全局梯度正则，bool类型（默认值：False）。

## 输入说明

params为参数的可迭代对象或参数组的dict类型。

## 输出说明

类型为“NpuFusedLamb”的对象。

## 异常说明

- “ValueError”- “lr”值小于0。
- “ValueError”- “betas”的值小于0或者betas的值大于1。
- “ValueError”- “eps”小于0。

## 约束说明

NpuFusedLamb的实现机制要求params中的每一个模型参数对象在使用过程中不能被重新申请，否则将导致无法预料的结果。引起模型参数对象被重新申请的操作包括但不限于：

- 将模型或其子Module进行.cpu操作
- 将模型参数对象指向新的对象
- 将模型参数对象置为None

对模型参数对象进行inplace计算，或者读取参数的值，NpuFusedLamb可正常工作。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

```python
import torch
from torch_npu.npu.amp import GradScaler, autocast
from torch_npu.optim import NpuFusedLamb 

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

opt_kwargs = dict(lr=0.01, eps=1e-5)
params = _create_simple_params_and_grads()
fused_opt = NpuFusedLamb(params, **opt_kwargs)
with torch.no_grad():
    fused_opt.step()
```

