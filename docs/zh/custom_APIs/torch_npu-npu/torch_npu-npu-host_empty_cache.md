# torch_npu.npu.host_empty_cache

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2 推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id7 -->

## 功能说明

释放当前由缓存持有的所有未占用的host物理内存。

## 定义文件

torch_npu/npu/memory.py

## 函数原型

```python
torch_npu.npu.host_empty_cache()
```

## 参数说明

无

## 返回值说明

无

## 约束说明

无

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> x = torch.empty([1024, 1024]).pin_memory()
>>> del x
>>> torch_npu.npu.host_empty_cache()
>>> print(torch_npu.npu.host_memory_stats())
```
