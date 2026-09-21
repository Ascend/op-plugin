# （beta）torch_npu.npu.utils.npu_check_overflow

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

检测梯度是否溢出。在INF_NAN模式下，检测输入`tensor`是否溢出；在饱和模式下，通过检查硬件溢出标志位判断是否溢出。

## 函数原型

```python
torch_npu.npu.utils.npu_check_overflow(grad) -> bool
```

## 参数说明

**grad**（`Tensor`或`float`）：在INF_NAN模式下判断输入中是否有`inf`或`nan`；饱和模式下，忽略输入，检查硬件溢出标志位。

## 返回值说明

`bool`

True表示溢出，False表示未溢出。

## 调用示例

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
