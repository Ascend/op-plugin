# torch_npu.profiler.supported_profiler_level

## 产品支持情况

| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品</term> |    √     |
| <term>Atlas 训练系列产品</term>    |    √     |

## 功能说明

查询当前支持的torch_npu.profiler.ProfilerLevel级别。

## 函数原型

```
torch_npu.profiler.supported_profiler_level()
```

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

```python
import torch
import torch_npu

...

torch_npu.profiler.supported_profiler_level()
```