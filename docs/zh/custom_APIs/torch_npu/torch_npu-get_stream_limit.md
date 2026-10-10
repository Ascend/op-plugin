# torch.npu.get_stream_limit

> [!NOTICE]  
> 此接口在本版本中有变更，具体变更内容请参考《版本说明》中的“[接口变更说明](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/zh/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E)”。

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- 通过该接口，获取指定Stream的Device资源限制。
- 若没有调用`torch.npu.set_stream_limit`接口设置Device资源限制，则调用本接口获取到的Device资源优先级为：当前进程的Device资源限制（调用`torch.npu.set_device_limit`接口设置）> 硬件默认资源限制。
- 当前支持资源类型为Cube Core、Vector Core。

## 函数原型

```python
torch.npu.get_stream_limit(stream) ->Dict
```

## 参数说明

**stream** (`torch_npu.npu.Stream`)：必选参数，目标流对象。

## 返回值说明

`Dict`

返回`stream`的Cube和Vector核数。

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
