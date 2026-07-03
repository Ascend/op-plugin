# (beta) torch_npu.npu.utils.is_support_inf_nan

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

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
