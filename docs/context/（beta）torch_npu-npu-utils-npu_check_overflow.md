# （beta）torch_npu.npu.utils.npu_check_overflow

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |



## 功能说明

检测梯度是否溢出，INF_NAN模式下检测输入`Tensor`是否溢出；饱和模式通过检查硬件溢出标志位判断是否溢出。

## 函数原型

```
torch_npu.npu.utils.npu_check_overflow(grad) -> bool
```
## 参数说明

- 输入为**torch.Tensor**或**float**，在**INF_NAN**模式下判断输入中是否有`inf`或`nan`
- 饱和模式忽略输入检查硬件溢出标志位

## 返回值说明


True溢出，False未溢出。



## 调用示例

```python
import torch

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

