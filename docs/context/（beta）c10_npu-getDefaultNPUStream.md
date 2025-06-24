# （beta）c10_npu::getDefaultNPUStream

## 定义文件

torch_npu\csrc\core\npu\NPUStream.h

## 函数原型

```
c10_npu::NPUStream c10_npu::getDefaultNPUStream(c10::DeviceIndex device_index = -1)
```

## 功能说明

获取默认NPU流，返回值类型NPUStream，与c10::CUDA::CUDAStream c10::CUDA::getDefaultCUDAStream(c10::DeviceIndex  _device_index_  = -1)相同。

## 参数说明

device_index：DeviceIndex类型，获取流的NPU设备id。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

