# （beta）c10_npu::getDefaultNPUStream

## 定义文件

torch_npu\csrc\core\npu\NPUStream.h

## 函数原型

```cpp
c10_npu::NPUStream c10_npu::getDefaultNPUStream(c10::DeviceIndex device_index = -1)
```

## 功能说明

获取默认NPU流，返回值类型NPUStream，其功能和使用方式与c10::cuda::CUDAStream c10::cuda::getDefaultCUDAStream(c10::DeviceIndex device_index = -1)相同。

## 参数说明

device_index：DeviceIndex类型，获取流的NPU设备ID。默认值为-1，表示使用当前的NPU设备。

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
