# （beta）c10::npu::current_device

## 定义文件

torch_npu\csrc\libs\init_npu.h

## 函数原型

```cpp
c10::DeviceIndex c10::npu::current_device()
```

## 功能说明

获取当前NPU设备，返回值类型DeviceIndex，与c10::DeviceIndex c10::cuda::current_device()相同。

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
