# (beta) torch_npu.npu.utils.npu_check_overflow

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas training products</term>: Supported
<!-- end id5 -->

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
