# （beta）c10_npu::set_device

## 定义文件

torch_npu\csrc\core\npu\NPUFunctions.h

## 函数原型

```cpp
void c10_npu::set_device(c10::DeviceIndex device)
```

## 功能说明

NPU设备设置，与PyTorch的1.11.0版本中void c10::cuda::set_device(c10::DeviceIndex _device_)相同，与c10_npu::SetDevice主要区别是增加了错误检查。

## 参数说明

device：DeviceIndex类型，待设置的NPU设备ID。

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>
<!-- end id4 -->
