# （beta）c10_npu::getNPUStreamFromPool

## 定义文件

torch_npu\csrc\core\npu\NPUStream.h

## 函数原型

```cpp
c10_npu::NPUStream c10_npu::getNPUStreamFromPool(c10::DeviceIndex device = -1)
```

## 功能说明

从NPU流池中获得一条新流，流是从池中预先分配的，并以循环的方式获取。返回值类型NPUStream。

## 参数说明

device：DeviceIndex类型，获取流的NPU设备ID。

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
