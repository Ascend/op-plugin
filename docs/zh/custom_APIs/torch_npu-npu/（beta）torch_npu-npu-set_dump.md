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
torch_npu.npu.set_dump(path_to_json)
```

## 参数说明

 **path_to_json**：配置文件所在的路径，包含文件名，用户需根据实际情况配置。具体配置请参考《CANN Runtime运行时API》中“<a href="https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_0155.html">函数：set_dump</a>”章节。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>>
>>> # 1. 初始化Dump
>>> torch_npu.npu.init_dump()
>>>
>>> # 2. 指定Dump 配置文件路径
>>> torch_npu.npu.set_dump("/home/HwHiAiUser/dump.json")
>>>
>>> # 3. 执行模型推理（示例）
>>> # output = model(input_data)
>>>
>>> # 4. 结束Dump
>>> torch_npu.npu.finalize_dump()
```
