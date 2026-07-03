# torch_npu.optim.NpuFusedOptimizerBase

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas training products</term>                                      |    √     |

## Function

Provides the base class for tensor fusion optimizers. This class implements basic optimizer functions such as gradient zeroing and gradient updating. Users can inherit from this class to implement custom fused optimizers.

## Prototype

```python
class torch_npu.optim.NpuFusedOptimizerBase(params, default)
```

## Parameters

- **`params`** (`iterable`): Required. Model parameters or parameter groups.
- **`default`** (`dict`): Dictionary containing all other parameters.

## Return Values

Object of the `NpuFusedOptimizerBase` type.

## Constraints

`NpuFusedOptimizerBase` is a base class and cannot be used independently. Users must inherit from this subclass to implement a fused optimizer with specific functionalities.

## Example

```python
import math
import torch
import torch_npu
from torch_npu.optim.npu_fused_optim_base import NpuFusedOptimizerBase
from torch.optim.optimizer import required

LR_MIN = 0.0
MOMENTUM_MIN = 0.0
DAMPENING_DEFAULT = 0.0
WEIGHT_DECAY_MIN = 0.0


class NpuFusedSGD(NpuFusedOptimizerBase):
    def __init__(self,
                 params,
                 lr=required,
                 momentum=MOMENTUM_MIN,
                 dampening=DAMPENING_DEFAULT,
                 weight_decay=WEIGHT_DECAY_MIN,
                 nesterov=False):
        if lr is not required and lr < LR_MIN:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < MOMENTUM_MIN:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < WEIGHT_DECAY_MIN:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <= MOMENTUM_MIN
                         or not math.isclose(dampening, DAMPENING_DEFAULT, abs_tol=1e-15)):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        defaults = dict(lr=lr,
                        momentum=momentum,
                        dampening=dampening,
                        weight_decay=weight_decay,
                        nesterov=nesterov)
        self._momentum_buffer_already_in_state = False
        super(NpuFusedSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NpuFusedSGD, self).__setstate__(state)
        for group in self.param_groups:            
            group.setdefault('nesterov', False)
```
