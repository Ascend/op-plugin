# （beta）c10_npu::SetDevice

## 定义文件

torch_npu\csrc\core\npu\NPUFunctions.h

## 函数原型

```cpp
aclError c10_npu::SetDevice(c10::DeviceIndex device)
```

## 功能说明

NPU设备设置，用于指定当前线程使用的NPU设备，返回值类型aclError。

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
