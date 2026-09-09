# (beta) torch_npu.npu.utils.npu_check_overflow

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Checks whether the gradient overflows. In `INF_NAN` mode, this function checks whether the input `Tensor` has overflowed. In saturation mode, it determines whether overflow has occurred by checking the hardware overflow flag.

## Prototype

```python
torch_npu.npu.utils.npu_check_overflow(grad) -> bool
```

## Parameters

**`grad`** (`Tensor` or `float`): In `INF_NAN` mode, this function checks whether the input contains `inf` or `nan`. In saturation mode, it ignores the input and determines overflow status by checking the hardware overflow flag.

## Return Values

`bool`

`True` (an overflow has occurred) or `False` (no overflow).

## Example

```python
import torch
import torch_npu
import torch_npu.npu.utils as utils
from torch_npu.testing.testcase import TestCase, run_tests


class TestCheckOverFlow(TestCase):

    def test_check_over_flow(self):
        a = torch.Tensor([65535]).npu().half()
        a = a + a
        ret = utils.npu_check_overflow(a)
        self.assertTrue(ret)
 
 
if __name__ == "__main__":
    run_tests()
```
