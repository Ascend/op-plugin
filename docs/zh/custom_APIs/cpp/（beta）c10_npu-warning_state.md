# （beta）c10_npu::warning_state

## 定义文件

torch_npu\csrc\core\npu\NPUFunctions.h

## 函数原型

```cpp
c10_npu::WarningState& c10_npu::warning_state()
```

## 功能说明

获取当前运行时警告等级，返回值类型WarningState为枚举类，包含无警告L_DISABLED、警告L_WARN和报错L_ERROR，与1.11.0版本中WarningState& c10::cuda::warning_state()相同。

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
