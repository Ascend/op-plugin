# （beta）c10_npu::warning_state

## 定义文件

torch_npu\csrc\core\npu\NPUFunctions.h

## 函数原型

```
c10_npu::WarningState& c10_npu::warning_state()
```

## 功能说明

获取当前同步时警告等级，返回值类型WarningState为枚举类，包含无警告L_DISABLED、警告L_WARN和报错L_ERROR，与WarningState& c10::cuda::warning_state()相同。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

