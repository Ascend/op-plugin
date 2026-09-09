# torch_npu.npu.disable_deterministic_with_backward

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |
|<term>Atlas 200I/500 A2 inference products</term>                                      |    √     |

## Function

Disables deterministic computation. A deterministic algorithm ensures that the model produces identical outputs for identical inputs during each forward propagation.

## Prototype

```python
torch_npu.npu.disable_deterministic_with_backward(tensor) -> Tensor
```

## Parameters

**`tensor`** (`Tensor`): This API transmits data transparently without data processing. The supported data types and data formats match those available for PyTorch on each chip, with zero interface-level constraints.

## Return Values

`Tensor`

Computation result of `disable_deterministic_with_backward`.

## Constraints

- The input `tensor` must be a tensor variable that can be propagated through the training network and is associated with the output of the entire network. Otherwise, backward deterministic capability cannot be enabled.
- Graph mode is not supported.

## Example

Single-operator call:

```python
import unittest
import torch
import torch.nn as nn
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

class TorchairSt(unittest.TestCase):
    def test_enable_to_disable_deterministic_algorithms(self):
        target_dtype = torch.float16        
        class DeterministicModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = x + 1
                x = torch_npu.npu.enable_deterministic_with_backward(x)
                add4 = x + y
                add1 = sum(add4)
                add1 = torch_npu.npu.disable_deterministic_with_backward(add1)
                add6 = add1 + add1
                return add6

        device = torch.device("npu:0")
        model = DeterministicModel()
        npu_mode = model.to(device)
        
        ins1 = torch.ones((2, 2), requires_grad=True).to(target_dtype).npu()
        ins2 = torch.ones((2, 2), requires_grad=True).to(target_dtype).npu()
        output_data = npu_mode(ins1, ins2)
        self.assertEqual(False, torch.are_deterministic_algorithms_enabled())

        loss_fn = nn.MSELoss()
        target_data = torch.randn((1, 2), requires_grad=True).to(target_dtype).npu()
        loss = loss_fn(output_data, target_data)
        loss.backward()
        self.assertEqual(False, torch.are_deterministic_algorithms_enabled())

if __name__ == '__main__':    
    unittest.main()

# Expected output of the preceding code sample:
----------------------------------------------------------------------
Ran 1 test in 4.636s

OK
```
