# （beta）c10_npu::device_count

## 定义文件

torch_npu\csrc\core\npu\NPUFunctions.h

## 函数原型

```
c10::DeviceIndex c10_npu::device_count()
```

## 功能说明

NPU设备数量获取，返回值类型DeviceIndex，与c10::DeviceIndex c10::cuda::device_count()相同。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

