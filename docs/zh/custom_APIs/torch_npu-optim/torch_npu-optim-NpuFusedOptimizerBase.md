# torch_npu.optim.NpuFusedOptimizerBase

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

张量融合优化器的基类，实现梯度清零、梯度更新等优化器基本功能，用户可进行继承实现自定义融合优化器。

## 函数原型

```
class torch_npu.optim.NpuFusedOptimizerBase(params, default)
```

## 参数说明

- **params** (`iterable`)：必选参数，模型参数或模型参数组。
- **default** (`dict`)：包含其他所有参数的字典。


## 返回值说明

类型为`NpuFusedOptimizerBase`的对象。

## 约束说明

`NpuFusedOptimizerBase`为基类，无法单独使用，需通过继承子类实现特定功能的融合优化器。


## 调用示例

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

