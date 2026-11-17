# 添加二进制黑名单示例

本文档以单个算子和多个算子两种场景为例，说明通过torch.npu.set_option设置“NPU_FUZZY_COMPILE_BLACKLIST”选项添加二进制黑名单的方法。

单个算子使用示例：

```python
import torch
import torch_npu

option = {}
option['NPU_FUZZY_COMPILE_BLACKLIST'] = "DynamicGRUV2"       #根据实际场景进行替换
torch.npu.set_option(option)
```

多个算子使用示例：

```python
import torch
import torch_npu

option = {}
option['NPU_FUZZY_COMPILE_BLACKLIST'] = "DynamicGRUV2,DynamicRNN"          #根据实际场景进行替换
torch.npu.set_option(option)
```
