# （beta）torch_npu::init_npu

## 定义文件

torch_npu\csrc\libs\init_npu.h

## 函数原型

```
void torch_npu::init_npu(const c10::DeviceIndex device_index = 0)
void torch_npu::init_npu(const std::string& device_str)
void torch_npu::init_npu(const at::Device& device)
```

## 功能说明

初始化NPU设备。

## 参数说明

- device_index：DeviceIndex类型，指定初始化的NPU设备id，默认0。
- device_str：string类型，指定初始化的设备名称。
- device：Device类型，指定初始化的NPU设备。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

