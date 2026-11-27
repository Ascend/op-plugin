# （beta）struct c10_npu::NPUHooksInterface

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

NPUHooksInterface是一个Hook接口类，提供了NPU Hook的相关接口。

## 定义文件

torch_npu/csrc/core/npu/NPUHooksInterface.h

## 成员函数

完整声明见定义文件`torch_npu/csrc/core/npu/NPUHooksInterface.h`。

- **const at::Generator& c10_npu::NPUHooksInterface::getDefaultGenerator(c10::DeviceIndex device_index) const**

获取指定NPU设备的默认随机数生成器。

**device_index** (`c10::DeviceIndex`)：必选参数。传入 `-1`时使用当前NPU设备。若显式指定设备ID，则须满足`0 <= device_index < c10_npu::device_count()`。
