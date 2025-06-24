# （beta）c10_npu::current_device

## 定义文件

torch_npu\csrc\core\npu\NPUFunctions.h

## 函数原型

```
c10::DeviceIndex c10_npu::current_device()
```

## 功能说明

NPU设备id获取，返回值类型DeviceIndex，表示获取到的设备id，与c10::DeviceIndex c10::cuda::current_device()相同，与c10_npu::GetDevice主要区别是增加了错误检查。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

