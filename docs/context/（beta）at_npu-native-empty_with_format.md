# （beta）at_npu::native::empty_with_format

## 定义文件

torch_npu\csrc\core\npu\NPUFormat.h

## 函数原型

```
at::Tensor at_npu::native::empty_with_format(c10::IntArrayRef sizes, const c10::TensorOptions& options, int64_t acl_format, bool keep_format = false)
```

## 功能说明

获取指定格式的NPU空tensor，返回值类型Tensor，表示获取的空tensor。

## 参数说明

sizes：IntArrayRef类型，获取tensor的size。

options：TensorOptions类型，获取tensor的可选信息，如dtype、device等。

acl_format：int64_t类型，指定获取的格式。

keep_format：bool类型，是否指定格式，true表示指定获取tensor的格式，false表示允许根据算子实际需求调整获取tensor的格式。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

