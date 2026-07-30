# （beta）c10_npu::SetDevice

## 定义文件

torch_npu\csrc\core\npu\NPUFunctions.h

## 函数原型

```cpp
aclError c10_npu::SetDevice(c10::DeviceIndex device)
```

## 功能说明

NPU设备设置，用于指定当前线程使用的NPU设备，返回值类型aclError。

## 参数说明

device：DeviceIndex类型，待设置的NPU设备ID。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>
