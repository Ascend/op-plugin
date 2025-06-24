# torch_npu.profiler.supported_activities

## 函数原型

```
torch_npu.profiler.supported_activities()
```

## 功能说明

查询当前支持采集的activities参数的CPU、NPU事件。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

```python
import torch
import torch_npu

...

torch_npu.profiler.supported_activities()
```

