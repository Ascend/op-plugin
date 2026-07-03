# torch_npu.optim.NpuFusedAdamW

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas training products</term>                                      |    √     |

## Function

Provides a high-performance AdamW (Adam with decoupled weight decay) optimizer implemented through tensor fusion. This optimizer is an enhanced version of Adam (Adaptive Moment Estimation). It decouples the weight decay term from gradient computation and applies it directly during parameter updates, addressing the interference between L2 regularization and adaptive learning rates in the original Adam algorithm. The core functionality is compatible with `torch.optim.AdamW`.

For details about the functions and principles of AdamW, see [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#adamw).

## Prototype

```python
class torch_npu.optim.NpuFusedAdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
```

## Parameters

- **`params`** (`iterable`): Required. Model parameters or parameter groups.
- **`lr`** (`float`): Optional. Learning rate. The default value is `1e-3`. A `ValueError` will be raised if this parameter is less than `0`.
- **`betas`** (`Tuple[float, float]`): Optional. Coefficients used for computing running averages of gradients and squared gradients. This parameter must be a tuple containing two values. The default value is `(0.9, 0.999)`. A `ValueError` will be raised if any value is less than `0` or greater than `1`.
- **`eps`** (`float`): Optional. Term added to the denominator to prevent division by 0 and improve numerical stability. The default value is `1e-8`. A `ValueError` will be raised if this parameter is less than `0`.
- **`weight_decay`** (`float`): Optional. Weight decay. The default value is `1e-2`. A `ValueError` will be raised if this parameter is less than `0`.
- **`amsgrad`** (`bool`): Optional. Specifies whether to use the AMSGrad variant described in [On the Convergence of Adam and Beyond](https://arxiv.org/pdf/1904.09237). The default value is `False`.

## Return Values

Object of the `NpuFusedAdamW` type.

## Constraints

The implementation mechanism of `NpuFusedAdamW` requires that each model parameter object inside `params` must not be reallocated during use. Otherwise, unpredictable results may occur. Operations that cause model parameter objects to be reallocated include, but are not limited to:

- Calling `.cpu()` on the model or its submodules
- Pointing a model parameter object to a new object
- Setting a model parameter object to `None`

This API supports normal execution when performing in-place computations on model parameter objects or reading parameter values.

## Example

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
