# （beta）c10_npu::GetDevice

## 定义文件

torch_npu\csrc\core\npu\NPUFunctions.h

## 函数原型

```cpp
aclError c10_npu::GetDevice(c10::DeviceIndex* device)
```

## 功能说明

NPU设备ID获取，返回值类型为aclError，与PyTorch 1.11.0版本中c10::cuda::GetDevice函数的返回值类型cudaError_t相同。

## 参数说明

device：DeviceIndex类型，存储获取的设备ID。

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
