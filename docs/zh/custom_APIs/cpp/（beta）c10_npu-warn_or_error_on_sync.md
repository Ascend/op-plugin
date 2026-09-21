# （beta）c10_npu::warn_or_error_on_sync

## 定义文件

torch_npu\csrc\core\npu\NPUFunctions.h

## 函数原型

```cpp
void c10_npu::warn_or_error_on_sync()
```

## 功能说明

NPU同步时警告，无返回值，根据当前警告等级进行报错或警告，与void c10::cuda::warn_or_error_on_sync()相同。

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas 训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 训练系列产品</term>
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas 推理系列产品</term>
<!-- end id4 -->
