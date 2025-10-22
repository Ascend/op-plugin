# torch.npu.get_stream_limit
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |


## 功能说明

- 通过该接口，获取指定Stream的Device资源限制。
- 若没有调用`torch.npu.set_stream_limit`接口设置Device资源限制，则调用本接口获取到的Device资源优先级为：当前进程的Device资源限制（调用`torch.npu.set_device_limit`接口设置）> 硬件默认资源限制。
- 当前支持资源类型为Cube Core、Vector Core。

## 函数原型

```
torch.npu.get_stream_limit(stream) ->Dict
```

## 参数说明

**stream** (`torch_npu.npu.Stream`)：必选参数，设置控核的流。

## 返回值说明
`Dict`

代表`stream`的Cube和Vector核数。

## 约束说明

无

## 调用示例

 ```python
>>> import torch
>>> import torch_npu

>>> torch.npu.set_stream_limit(torch.npu.current_stream(),12,20)
>>> print(torch.npu.get_stream_limit(torch.npu.current_stream()))
{"cube_core_num":12, "vector_core_num":20}
 ```
