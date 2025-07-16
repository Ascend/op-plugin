# （beta）at::Device

## 函数原型

```
at::Device(const std::string &device_string)
```

## 功能说明

在安装torch_npu后，Device类型新增支持NPU字段，可以从字符串描述中指示设备。

## 参数说明

device_ring：string类型，提供的字符串必须遵循以下架构：(npu)[:<device-index\>]，其中NPU指定设备类型，<device-index\>可选，指定设备索引。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

