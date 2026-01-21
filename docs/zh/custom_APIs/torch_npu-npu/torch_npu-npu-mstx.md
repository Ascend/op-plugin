# torch_npu.npu.mstx

## 产品支持情况

| 产品                                                      | 是否支持 |
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term>                        |    √     |
| <term>Atlas A3 推理系列产品</term>                        |    √     |
| <term>Atlas A2 训练系列产品</term>                        |    √     |
| <term>Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 推理系列产品</term>                           |    √     |
| <term>Atlas 训练系列产品</term>                           |    √     |

## 功能说明

打点接口。

用于为[torch_npu.profiler._ExperimentalConfig](torch_npu-profiler-_ExperimentalConfig.md)的mstx提供打点接口调用。

## 函数原型

```
torch_npu.npu.mstx()
```

## 参数说明

无

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。


```python
import torch
import torch_npu
mstx_object = torch_npu.npu.mstx()
```