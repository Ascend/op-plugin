# （beta）torch_npu.npu.utils.is_support_inf_nan
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

判断当前使用的溢出检测模式。

## 函数原型

```
torch_npu.npu.utils.is_support_inf_nan() -> bool
```


## 返回值说明
`bool`

返回值为True时，代表是INF_NAN模式。

返回值为False时，代表为饱和模式。


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

