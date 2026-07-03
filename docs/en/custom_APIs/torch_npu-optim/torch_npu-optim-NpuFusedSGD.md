# torch_npu.optim.NpuFusedSGD

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas training products</term>                                      |    √     |

## Function

Provides a high-performance SGD optimizer implemented through tensor fusion. The core functionality is compatible with `torch.optim.SGD`.

For details about the functions and principles of SGD, see [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#sgd).

## Prototype

```python
class torch_npu.optim.NpuFusedSGD(params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

## Parameters

- **`params`** (`iterable`): Required. Model parameters or parameter groups.
- **`lr`** (`float`): Required. Learning rate. A `ValueError` will be raised if this parameter is less than `0`.
- **`momentum`** (`float`): Optional. Momentum factor. The default value is `0`. A `ValueError` will be raised if this parameter is less than `0`.
- **`dampening`** (`float`): Optional. Momentum dampening factor. The default value is `0`.
- **`weight_decay`** (`float`): Optional. Weight decay. The default value is `0`. A `ValueError` will be raised if this parameter is less than `0`.
- **`nesterov`** (`bool`): Optional. Specifies whether to use Nesterov momentum. The default value is `False`. A `TypeError` will be raised if this parameter is set to `True` while `momentum` is less than `0` or `dampening` is not equal to `0`.

## Return Values

Object of the `NpuFusedSGD` type.

## Constraints

The implementation mechanism of `NpuFusedSGD` requires that each model parameter object inside `params` must not be reallocated during use. Otherwise, unpredictable results may occur. Operations that cause model parameter objects to be reallocated include, but are not limited to:

- Calling `.cpu()` on the model or its submodules
- Pointing a model parameter object to a new object
- Setting a model parameter object to `None`

This API supports normal execution when performing in-place computations on model parameter objects or reading parameter values.

## Example

```python
import torch
from torch_npu.npu.amp import GradScaler, autocast
from torch_npu.optim import NpuFusedSGD

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

opt_kwargs = dict(lr=0.01, momentum=0.9, weight_decay=0.001)
params = _create_simple_params_and_grads()
fused_opt = NpuFusedSGD(params, **opt_kwargs)
with torch.no_grad():
    fused_opt.step()
```
