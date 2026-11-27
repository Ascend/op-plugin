# 添加二进制黑名单示例

本文档以单个算子和多个算子两种场景为例，说明通过torch.npu.set_option设置“NPU_FUZZY_COMPILE_BLACKLIST”选项添加二进制黑名单的方法。

`NPU_FUZZY_COMPILE_BLACKLIST`接收字符串，取值为一个算子名称或以英文逗号分隔的多个算子名称。列表初始为空。请在运行目标算子前设置该选项。全局关闭即时编译时，命中列表的算子仍走即时编译路径。

以下示例将`DynamicGRUV2`加入黑名单。

```python
import torch
import torch_npu

option = {}
option['NPU_FUZZY_COMPILE_BLACKLIST'] = "DynamicGRUV2"
torch.npu.set_option(option)
```

以下示例一次将`DynamicGRUV2`和`DynamicRNN`加入黑名单。

```python
import torch
import torch_npu

option = {}
option['NPU_FUZZY_COMPILE_BLACKLIST'] = "DynamicGRUV2,DynamicRNN"
torch.npu.set_option(option)
```
