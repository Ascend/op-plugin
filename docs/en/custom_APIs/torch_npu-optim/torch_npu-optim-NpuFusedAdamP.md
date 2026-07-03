# torch_npu.optim.NpuFusedAdamP

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas training products</term>                                      |    √     |

## Function

Provides a high-performance AdamP optimizer implemented through tensor fusion. For details about the functions and principles of AdamP, see [AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights](https://arxiv.org/pdf/2006.08217).

## Prototype

```python
class torch_npu.optim.NpuFusedAdamP(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False)
```

## Parameters

- **`params`** (`iterable`): Required. Model parameters or parameter groups.
- **`lr`** (`float`): Optional. Learning rate. The default value is `1e-3`.
- **`betas`** (`Tuple[float, float]`): Optional. Coefficients used for computing running averages of gradients and squared gradients. This parameter must be a tuple containing two values. The default value is `(0.9, 0.999)`.
- **`eps`** (`float`): Optional. Term added to the denominator to prevent division by 0 and improve numerical stability. The default value is `1e-8`.
- **`weight_decay`** (`float`): Optional. Weight decay. The default value is `0`.
- **`delta`** (`float`): Optional. Cosine similarity threshold. The default value is `0.1`.
- **`wd_ratio`** (`float`): Optional. Dynamic adjustment rate for weight decay. The default value is `0.1`.
- **`nesterov`** (`bool`): Optional. Specifies whether to use Nesterov momentum. The default value is `False`.

## Return Values

Object of the `NpuFusedAdamP` type.

## Constraints

The implementation mechanism of `NpuFusedAdamP` requires that each model parameter object inside `params` must not be reallocated during use. Otherwise, unpredictable results may occur. Operations that cause model parameter objects to be reallocated include, but are not limited to:

- Calling `.cpu()` on the model or its submodules
- Pointing a model parameter object to a new object
- Setting a model parameter object to `None`

This API supports normal execution when performing in-place computations on model parameter objects or reading parameter values.

## Example

```python
import torch
from torch_npu.npu.amp import GradScaler, autocast
from torch_npu.optim import NpuFusedAdamP 

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

opt_kwargs = dict(eps=1e-5, betas=(0.9, 0.999), lr=2e-3, weight_decay=0.05)
params = _create_simple_params_and_grads()
fused_opt = NpuFusedAdamP(params, **opt_kwargs)
with torch.no_grad():
    fused_opt.step()
```
