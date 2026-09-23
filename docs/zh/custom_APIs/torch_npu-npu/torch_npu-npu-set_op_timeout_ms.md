# torch_npu.npu.set_op_timeout_ms

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id6 -->

## 功能说明

该接口用于设置NPU上算子的执行超时时间，单位为毫秒（ms）。

## 函数原型

```python
torch_npu.npu.set_op_timeout_ms(timeout)
```

## 参数说明

**timeout**（`int`）：根据传入的timeout值，设置算子的执行超时时间，单位为毫秒（ms）。

## 返回值说明

无

## 约束说明

无

## 调用示例

```python
import torch
import torch_npu

torch_npu.npu.set_op_timeout_ms(1000)
```
