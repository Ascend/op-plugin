# torch_npu.optim.NpuFusedAdadelta

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas training products</term>                                      |    √     |

## Function

Provides a high-performance Adadelta optimizer implemented through tensor fusion. The core functionality is compatible with `torch.optim.Adadelta`.

For details about the functions and principles of Adadelta, see [Adadelta](https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#adadelta).

## Prototype

```python
class torch_npu.optim.NpuFusedAdadelta(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)
```

## Parameters

- **`params`** (`iterable`): Required. Model parameters or parameter groups.
- **`lr`** (`float`): Optional. Learning rate. The default value is `1.0`. A `ValueError` will be raised if this parameter is less than `0`.
- **`rho`** (`float`): Optional. Coefficient used for computing a running average of squared gradients. The default value is `0.9`. A `ValueError` will be raised if this parameter is less than `0` or greater than `1`.
- **`eps`** (`float`): Optional. Term added to the denominator to prevent division by 0 and improve numerical stability. The default value is `1e-6`. A `ValueError` will be raised if this parameter is less than `0`.
- **`weight_decay`** (`float`): Optional. Weight decay. The default value is `0`. A `ValueError` will be raised if this parameter is less than `0`.

## Return Values

Object of the `NpuFusedAdadelta` type.

## Constraints

The implementation mechanism of `NpuFusedAdadelta` requires that each model parameter object in `params` must not be reallocated during use. Otherwise, unpredictable results may occur. Operations that cause model parameter objects to be reallocated include, but are not limited to:

- Calling `.cpu()` on the model or its submodules
- Pointing a model parameter object to a new object
- Setting a model parameter object to `None`

This API supports normal execution when performing in-place computations on model parameter objects or reading parameter values.

## Example

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
