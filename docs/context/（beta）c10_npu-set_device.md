# （beta）c10_npu::set_device

## 定义文件

torch_npu\csrc\core\npu\NPUFunctions.h

## 函数原型

```
void c10_npu::set_device(c10::DeviceIndex device)
```

## 功能说明

NPU设备设置，与void c10::cuda::set_device(c10::DeviceIndex  _device_)相同，与c10_npu::SetDevice主要区别是增加了错误检查。

## 参数说明

device：DeviceIndex类型，待设置的NPU设备id。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

