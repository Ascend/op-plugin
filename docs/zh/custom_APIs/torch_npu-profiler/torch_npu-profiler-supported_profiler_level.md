# torch_npu.profiler.supported_profiler_level

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

查询当前支持的torch_npu.profiler.ProfilerLevel级别。

## 函数原型

```python
torch_npu.profiler.supported_profiler_level()
```

## 返回值说明

返回{'Level0', 'Level1', 'Level_none', 'Level2'}则表示成功；无返回则表示失败。

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝运行，仅供参考。

```python
import torch
import torch_npu

...

torch_npu.profiler.supported_profiler_level()
```
