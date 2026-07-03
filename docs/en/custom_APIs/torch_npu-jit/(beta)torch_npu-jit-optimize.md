# (beta) torch_npu.jit.optimize

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Optimizes a ScriptFunction or ScriptModule for better performance.

## Prototype

```python
torch_npu.jit.optimize(jit_mod)
```

## Parameters

**`jit_mod`**: Required. ScriptFunction or ScriptModule to be optimized.

## Example

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
