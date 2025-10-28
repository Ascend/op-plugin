# torch_npu.optim.NpuFusedBertAdam

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

通过张量融合实现的高性能BertAdam优化器。BertAdam的功能和原理可参考[https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/optimization.py\#L64](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/optimization.py#L64)。

## 函数原型

```
class torch_npu.optim.NpuFusedBertAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
```


## 参数说明

- **params** (`dict`)：必选参数，模型参数或模型参数组，`params`为参数的可迭代对象或参数组的dict类型。
- **lr** (`float`)：可选参数，学习率，默认值为1e-3。`lr`的值小于0时，打印“ValueError”异常信息。
- **warmup** (`float`)：`t_total`的warmup比例，默认值为-1，表示不进行warmup。`warmup`的值小于0且`warmup`不等于-1，或者`warmup`大于等于1，打印“ValueError”异常信息。
- **t_total** (`float`)：学习率调整的步数，默认值为-1，表示固定学习率。
- **schedule** (`str`)：学习率warmup策略，默认值为warmup_linear。`schedule`为字符串，其值必须为warmup_cosine、warmup_constant、warmup_linear、warmup_poly中的一个。
- **b1** (`float`)：Adam b1，默认值为0.9。`b1`的值小于0或大于1时，打印“ValueError”异常信息。
- **b2** (`float`)：Adam b2，默认值为0.99。`b2`的值小于0或大于1时，打印“ValueError”异常信息。
- **e** (`float`)：Adam epsilon，默认值为1e-6。`e`的值小于0时，打印“ValueError”异常信息。
- **weight_decay** (`float`)：可选参数，权重衰减，默认值为0.01。
- **max_grad_norm** (`float`)：最大梯度范围，默认值为1.0，-1表示不做裁剪。


## 返回值说明

类型为`NpuFusedBertAdam`的对象。


## 约束说明

`NpuFusedBertAdam`的实现机制要求`params`中的每一个模型参数对象在使用过程中不能被重新申请，否则将导致无法预料的结果。引起模型参数对象被重新申请的操作包括但不限于：

- 将模型或其子Module进行.cpu操作
- 将模型参数对象指向新的对象
- 将模型参数对象置为None

对模型参数对象进行inplace计算，或者读取参数的值，`NpuFusedBertAdam`可正常工作。


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

