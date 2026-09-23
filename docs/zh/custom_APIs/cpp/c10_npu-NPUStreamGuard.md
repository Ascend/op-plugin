# c10_npu::NPUStreamGuard

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->

## 功能说明

NPU设备流guard，保障作用域内的设备流，与`c10::cuda::CUDAStreamGuard`相同。

## 定义文件

torch_npu\csrc\core\npu\NPUGuard.h

## 函数原型

```cpp
struct c10_npu::NPUStreamGuard
```
