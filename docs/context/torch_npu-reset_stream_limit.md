# torch.npu.reset_stream_limit
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |


## 功能说明

调用`torch.npu.set_stream_limit`接口设置指定Stream的Device资源限制后，可调用本接口重置指定Stream的Device资源限制，恢复默认配置，此时可通过`torch.npu.get_stream_limit`接口查询默认的资源设置。

## 函数原型

```
torch.npu.reset_stream_limit(stream) -> None
```

## 参数说明

**stream** (`torch_npu.npu.Stream`)：必选参数，设置控核的流。

## 返回值说明
`None`

代表无返回值。

## 约束说明

无

## 调用示例

 ```python
>>> import torch
>>> import torch_npu

>>> torch.npu.set_stream_limit(torch.npu.current_stream(), 12, 24)
>>> torch.npu.reset_stream_limit(torch.npu.Stream())
 ```
