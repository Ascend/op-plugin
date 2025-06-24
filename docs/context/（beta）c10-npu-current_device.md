# （beta）c10::npu::current_device

## 定义文件

torch_npu\csrc\libs\init_npu.h

## 函数原型

```
c10::DeviceIndex c10::npu::current_device()
```

## 功能说明

获取当前NPU设备，返回值类型DeviceIndex，与c10::DeviceIndex c10::cuda::current_device()相同。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

