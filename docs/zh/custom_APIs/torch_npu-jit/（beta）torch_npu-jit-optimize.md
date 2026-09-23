# （beta）torch_npu.jit.optimize

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id5 -->

## 功能说明

主要用于优化ScriptFunction或ScriptModule，以获取更好的性能。

## 函数原型

```python
torch_npu.jit.optimize(jit_mod)
```

## 参数说明

**jit_mod**：必选参数。用于被优化的ScriptFunction或ScriptModule。

## 调用示例

```python
import torch
import torch_npu
from torch_npu import jit

class SimpleModel(torch.nn.Module):
    def forward(self, x, y):
        z = x + y
        return torch.relu(z)

model = SimpleModel().eval()
traced_model = torch.jit.trace(model, (torch.rand(1, 3), torch.rand(1, 3)))

torch_npu.jit.optimize(traced_model)

```
