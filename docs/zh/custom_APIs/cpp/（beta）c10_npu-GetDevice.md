# （beta）c10_npu::GetDevice

## 产品支持情况

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->

## 功能说明

NPU设备ID获取，返回值类型为aclError，与PyTorch 1.11.0版本中c10::cuda::GetDevice函数的返回值类型cudaError_t相同。

## 定义文件

torch_npu/csrc/core/npu/NPUFunctions.h

## 函数原型

```cpp
aclError c10_npu::GetDevice(int32_t* device)
```

## 参数说明

device：int32_t类型的指针，输出参数，调用前需预分配用于存储设备ID的内存空间，返回获取到的设备ID。
