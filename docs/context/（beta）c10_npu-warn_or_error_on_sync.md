# （beta）c10_npu::warn_or_error_on_sync

## 定义文件

torch_npu\csrc\core\npu\NPUFunctions.h

## 函数原型

```
void c10_npu::warn_or_error_on_sync()
```

## 功能说明

NPU同步时警告，无返回值，根据当前警告等级进行报错或警告，与void c10::cuda::warn_or_error_on_sync()相同。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

