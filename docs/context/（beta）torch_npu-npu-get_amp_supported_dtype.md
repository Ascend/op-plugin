# （beta）torch_npu.npu.get_amp_supported_dtype

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |



## 功能说明

获取npu设备支持的数据类型，可能设备支持不止一种数据类型。

## 函数原型

```
torch_npu.npu.get_amp_supported_dtype()
```

## 返回值说明

**List**(`torch.dtype`)

## 调用示例

```python
import torch
import torch_npu

supported_dtypes = torch_npu.npu.get_amp_supported_dtype()
print(f"NPU支持的AMP数据类型：{supported_dtypes}")

```

