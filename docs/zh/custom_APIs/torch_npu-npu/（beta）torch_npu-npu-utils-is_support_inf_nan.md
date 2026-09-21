# （beta）torch_npu.npu.utils.is_support_inf_nan

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id5 -->

## 功能说明

判断当前使用的溢出检测模式。

## 函数原型

```python
torch_npu.npu.utils.is_support_inf_nan() -> bool
```

## 返回值说明

`bool`

返回值为True时，代表INF_NAN模式。

返回值为False时，代表饱和模式。

## 调用示例

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
