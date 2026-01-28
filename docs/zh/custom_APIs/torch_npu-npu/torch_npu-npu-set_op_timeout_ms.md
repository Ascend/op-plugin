# torch_npu.npu.set_op_timeout_ms

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品 / Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品 / Atlas A2 推理系列产品</term>  |    √     |
|<term>Atlas 200I / 500 A2 推理产品</term> |    √     |
|<term>Atlas 推理系列产品</term> |    √     |
|<term>Atlas 训练系列产品</term> |    √     |

## 功能说明

该接口用于设置NPU上算子的执行超时时间，单位为毫秒（ms）。

## 函数原型

```
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