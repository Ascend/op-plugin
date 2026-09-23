# （beta）at_npu::native::empty_with_format

## 定义文件

torch_npu\csrc\core\npu\NPUFormat.h

## 函数原型

```cpp
at::Tensor at_npu::native::empty_with_format(c10::IntArrayRef sizes, const c10::TensorOptions& options, int64_t acl_format, bool keep_format = false)
```

## 功能说明

获取指定格式的NPU空tensor，返回值类型Tensor，表示获取的空tensor。

## 参数说明

- sizes：IntArrayRef类型，指定tensor的维度。

- options：TensorOptions类型，指定tensor的可选信息，如dtype、device等。

- acl_format：int64_t类型，指定tensor的内存格式。

- keep_format：bool类型，是否保持指定格式，true表示固定tensor的格式，false表示允许根据算子实际需求调整tensor的格式。

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
