# （beta）c10_npu::getNPUStreamFromPool

## 定义文件

torch_npu\csrc\core\npu\NPUStream.h

## 函数原型

```
c10_npu::NPUStream c10_npu::getNPUStreamFromPool(c10::DeviceIndex device = -1)
```

## 功能说明

从NPU流池中获得一条新流，流是从池中预先分配的，并以循环的方式获取。返回值类型NPUStream。

## 参数说明

device：DeviceIndex类型，获取流的NPU设备id。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

