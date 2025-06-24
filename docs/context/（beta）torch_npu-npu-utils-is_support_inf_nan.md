# （beta）torch_npu.npu.utils.is_support_inf_nan

## 函数原型

```
torch_npu.npu.utils.is_support_inf_nan() -> bool
```

## 功能说明

判断当前使用的溢出检测模式，True为INF_NAN模式，False为饱和模式。

## 输出说明

True为INF_NAN模式，False为饱和模式。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

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

