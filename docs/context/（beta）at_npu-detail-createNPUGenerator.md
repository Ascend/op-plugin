# （beta）at_npu::detail::createNPUGenerator

## 定义文件

torch_npu\csrc\aten\NPUGeneratorImpl.h

## 函数原型

```
at::Generator at_npu::detail::createNPUGenerator(c10::DeviceIndex device_index = -1)
```

## 功能说明

NPU设备默认生成器创建，返回值类型Generator，与at::Generator at::cuda::detail::createCUDAGenerator(c10::DeviceIndex  _device_index_  = -1)相同。

## 参数说明

device_index：DeviceIndex类型，指定创建生成器的NPU设备id。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

