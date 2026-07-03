# torch_npu.optim.NpuFusedRMSpropTF

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas training products</term>                                      |    √     |

## Function

Provides a high-performance RMSpropTF optimizer implemented through tensor fusion. RMSpropTF is an RMSprop variant provided in the timm (pytorch-image-models) library. Its implementation is aligned with the behavior of TensorFlow-style RMSprop. It differs from `torch.optim.RMSprop` in initialization methods, epsilon placement, and momentum update order. The core functionality is compatible with `torch.optim.RMSprop`. For details about the functions and principles of RMSpropTF, see [https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/rmsprop_tf.py\#L14](https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/rmsprop_tf.py#L14).

## Prototype

```python
class torch_npu.optim.NpuFusedRMSpropTF(params, lr=1e-2, alpha=0.9, eps=1e-10, weight_decay=0, momentum=0., centered=False, decoupled_decay=False, lr_in_momentum=True)
```

## Parameters

- **`params`** (`iterable`): Required. Model parameters or parameter groups.
- **`lr`** (`float`): Optional. Learning rate. The default value is `1e-2`. A `ValueError` will be raised if this parameter is less than `0`.
- **`alpha`** (`float`): Optional. Smoothing constant. The default value is `0.9`. A `ValueError` will be raised if this parameter is less than `0`.
- **`eps`** (`float`): Optional. Term added to the denominator to prevent division by 0 and improve numerical stability. The default value is `1e-10`. A `ValueError` will be raised if this parameter is less than `0`.
- **`weight_decay`** (`float`): Optional. Weight decay. The default value is `0`. A `ValueError` will be raised if this parameter is less than `0`.
- **`momentum`** (`float`): Optional. Momentum factor. The default value is `0`. A `ValueError` will be raised if this parameter is less than `0`.
- **`centered`** (`bool`): Optional. Specifies whether to compute the centered RMSProp where the gradient is normalized by an estimation of its variance. The default value is `False`.
- **`decoupled_decay`** (`bool`): Optional. Specifies whether to apply weight decay only to parameters. The default value is `False`.
- **`lr_in_momentum`** (`bool`): Optional. Specifies whether to use the learning rate when computing the momentum buffer. The default value is `True`.

## Return Values

Object of the `NpuFusedRMSpropTF` type.

## Constraints

The implementation mechanism of `NpuFusedRMSpropTF` requires that each model parameter object inside `params` must not be reallocated during use. Otherwise, unpredictable results may occur. Operations that cause model parameter objects to be reallocated include, but are not limited to:

- Calling `.cpu()` on the model or its submodules
- Pointing a model parameter object to a new object
- Setting a model parameter object to `None`

This API supports normal execution when performing in-place computations on model parameter objects or reading parameter values.

## Example

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
