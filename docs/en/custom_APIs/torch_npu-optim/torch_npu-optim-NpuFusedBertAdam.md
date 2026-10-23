# torch_npu.optim.NpuFusedBertAdam

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->

## Function

Provides a high-performance BertAdam optimizer implemented through tensor fusion. For details about the functions and principles of BertAdam, see [https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/optimization.py\#L64](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/optimization.py#L64).

## Prototype

```python
class torch_npu.optim.NpuFusedBertAdam(params, lr=1e-3, warmup=-1, t_total=-1, schedule="warmup_linear", b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0)
```

## Parameters

- **`params`** (`iterable`): Required. Model parameters or parameter groups.
- **`lr`** (`float`): Optional. Learning rate. The default value is `1e-3`. A `ValueError` will be raised if this parameter is less than `0`.
- **`warmup`** (`float`): Optional. Warmup ratio of `t_total`. The default value is `-1`, indicating that warmup is disabled. A `ValueError` will be raised if this parameter is less than `0` and not equal to `-1`, or if it is greater than or equal to `1`.
- **`t_total`** (`float`): Optional. Number of steps for learning rate scheduling. The default value is `-1`, indicating that a fixed learning rate is used.
- **`schedule`** (`str`): Optional. Learning rate warmup strategy. The default value is `"warmup_linear"`. `schedule` is a string. Valid values are `"warmup_cosine"`, `"warmup_constant"`, `"warmup_linear"`, or `"warmup_poly"`.
- **`b1`** (`float`): Optional. Adam coefficient `b1`. The default value is `0.9`. A `ValueError` will be raised if this parameter is less than `0` or greater than or equal to `1`.
- **`b2`** (`float`): Optional. Adam coefficient `b2`. The default value is `0.999`. A `ValueError` will be raised if this parameter is less than `0` or greater than or equal to `1`.
- **`e`** (`float`): Optional. Adam epsilon value. The default value is `1e-6`. A `ValueError` will be raised if this parameter is less than `0`.
- **`weight_decay`** (`float`): Optional. Weight decay. The default value is `0.01`.
- **`max_grad_norm`** (`float`): Optional. Maximum gradient norm. The default value is `1.0`. A value of `-1` indicates that gradient clipping is disabled.

## Return Values

Object of the `NpuFusedBertAdam` type.

## Constraints

The implementation mechanism of `NpuFusedBertAdam` requires that each model parameter object inside `params` must not be reallocated during use. Otherwise, unpredictable results may occur. Operations that cause model parameter objects to be reallocated include, but are not limited to:

- Calling `.cpu()` on the model or its submodules
- Pointing a model parameter object to a new object
- Setting a model parameter object to `None`

This API supports normal execution when performing in-place computations on model parameter objects or reading parameter values.

## Example

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
