# （beta）torch_npu.npu.is_jit_compile_false

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

确认JIT编译模式是否被禁用，如果被禁用，返回True，否则返回False。

<term>Ascend 950DT系列产品</term>仅返回True，即JIT编译模式默认禁用。

## 函数原型

```python
torch_npu.npu.is_jit_compile_false()
```

## 返回值说明

bool型。

## 调用示例

```python
import torch
import torch_npu
torch_npu.npu.set_compile_mode(jit_compile=False)
print(torch_npu.npu.is_jit_compile_false())
True
```
