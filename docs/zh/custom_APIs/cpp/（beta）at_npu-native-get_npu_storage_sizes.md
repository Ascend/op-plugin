# （beta）at_npu::native::get_npu_storage_sizes

## 定义文件

torch_npu\csrc\core\npu\NPUFormat.h

## 函数原型

```cpp
std::vector<int64_t> at_npu::native::get_npu_storage_sizes(const at::Tensor& self)
```

## 功能说明

获取NPU tensor存储的各维度大小，返回值类型vector\<int64_t>，表示NPU tensor底层存储每一维的大小。

## 参数说明

self：Tensor类型，待获取存储各维度大小的tensor。

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
