# （beta）struct c10_npu::NPUHooksInterface

## 定义文件

torch_npu\csrc\core\npu\NPUHooksInterface.h

## 功能说明

NPUHooksInterface是一个Hook接口类，提供了NPU Hook的相关接口。

## 成员函数

**const at::Generator& c10_npu::NPUHooksInterface::getDefaultGenerator(c10::DeviceIndex**

**device_index)**

NPUHooksInterface获取默认随机数生成器，与const at::Generator& at::CUDAHooksInterface::getDefaultCUDAGenerator(c10::DeviceIndex  _device_index_  = -1)相同。

device_index：DeviceIndex类型，指定NPU设备id。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

