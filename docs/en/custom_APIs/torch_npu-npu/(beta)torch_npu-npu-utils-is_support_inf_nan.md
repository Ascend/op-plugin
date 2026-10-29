# (beta) torch_npu.npu.utils.is_support_inf_nan

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

Determines the current overflow detection mode.

## Prototype

```python
torch_npu.npu.utils.is_support_inf_nan() -> bool
```

## Return Values

`bool`

`True` indicates `INF_NAN` mode.

`False` indicates saturation mode.

## Example

```python
import torch
 
import torch_npu.npu.utils as utils
from torch_npu.testing.testcase import TestCase, run_tests
 
 
class TestCheckOverFlow(TestCase):
 
    def test_check_over_flow(self):
        ret = utils.is_support_inf_nan()
        self.assertTrue(ret)
 
 
if __name__ == "__main__":
    run_tests()
```
