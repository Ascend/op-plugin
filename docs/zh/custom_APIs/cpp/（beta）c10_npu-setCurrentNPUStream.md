# （beta）c10_npu::setCurrentNPUStream

## 定义文件

torch_npu\csrc\core\npu\NPUStream.h

## 函数原型

```cpp
void c10_npu::setCurrentNPUStream(c10_npu::NPUStream stream)
```

## 功能说明

设置当前NPU流，与void c10::cuda::setCurrentCUDAStream(c10::cuda::CUDAStream _stream_)相同。

## 参数说明

stream：NPUStream类型，待设置的NPU流。

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
