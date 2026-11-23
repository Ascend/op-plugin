# （beta）torch_npu.npu.set_dump

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
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id5 -->

## 功能说明

传入配置文件来配置dump参数。

## 函数原型

```python
torch_npu.npu.set_dump(cfg_file)
```

## 参数说明

**cfg_file**（`str`）：必填参数，指定Dump配置文件的路径，需包含文件名。配置项请参考《CANN Runtime运行时API》的[set_dump](https://gitcode.com/cann/runtime/blob/9.2.0/python/docs/zh/api_ref/runtime/dump/function-set_dump.md)。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>>
>>> # 1. 初始化Dump
>>> torch_npu.npu.init_dump()
>>>
>>> # 2. 指定Dump配置文件路径
>>> torch_npu.npu.set_dump("/home/HwHiAiUser/dump.json")
>>>
>>> # 3. 执行模型推理（示例）
>>> # output = model(input_data)
>>>
>>> # 4. 结束Dump
>>> torch_npu.npu.finalize_dump()
```
