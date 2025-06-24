# torch_npu.optim.NpuFusedBertAdam

## 函数原型

```
class torch_npu.optim.NpuFusedBertAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
```

## 功能说明

通过张量融合实现的高性能BertAdam优化器。BertAdam的功能和原理可参考[https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/optimization.py\#L64](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/optimization.py#L64)。

## 参数说明

- params：模型参数或模型参数组。
- lr：学习率，float类型（默认值：1e-3）。
- warmup：t_total的warmup比例（默认值：-1，表示不进行warmup）。
- t_total：学习率调整的步数，float类型（默认值：-1，表示固定学习率）。
- schedule：学习率warmup策略，str类型（默认值：'warmup_linear'）。
- b1：Adam b1，float类型（默认值：0.9）。
- b2：Adam b2，float类型（默认值：0.99）。
- e：Adam epsilon，float类型（默认值：1e-6）。
- weight_decay：权重衰减，float类型（默认值：0.01）。
- max_grad_norm：最大梯度正则，float类型（默认值：1.0，-1表示不做裁剪）。

## 输入说明

params为参数的可迭代对象或参数组的dict类型。schedule为字符串，其值必须为warmup_cosine、warmup_constant、warmup_linear、warmup_poly中的一个。

## 输出说明

类型为“NpuFusedBertAdam”的对象。

## 异常说明

- “ValueError”- “lr”值小于0。
- “ValueError”- “warmup”的值小于0且warmup不等于-1，或者warmup大于等于1。
- “ValueError”- “b1”的值小于0或大于1。
- “ValueError”- “b2”的值小于0或大于1。
- “ValueError”- “e”的值小于0。

## 约束说明

NpuFusedBertAdam的实现机制要求params中的每一个模型参数对象在使用过程中不能被重新申请，否则将导致无法预料的结果。引起模型参数对象被重新申请的操作包括但不限于：

- 将模型或其子Module进行.cpu操作
- 将模型参数对象指向新的对象
- 将模型参数对象置为None

对模型参数对象进行inplace计算，或者读取参数的值，NpuFusedBertAdam可正常工作。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

```python
import torch
from torch_npu.npu.amp import GradScaler, autocast
from torch_npu.optim import NpuFusedBertAdam 

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

opt_kwargs = dict(lr=0.01, warmup=0.1, t_total=20, max_grad_norm=-1)
params = _create_simple_params_and_grads()
fused_opt = NpuFusedBertAdam(params, **opt_kwargs)
with torch.no_grad():
    fused_opt.step()
```

