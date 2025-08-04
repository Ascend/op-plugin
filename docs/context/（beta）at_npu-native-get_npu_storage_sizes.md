# （beta）at_npu::native::get_npu_storage_sizes

## 定义文件

torch_npu\csrc\core\npu\NPUFormat.h

## 函数原型

```
std::vector<int64_t> at_npu::native::get_npu_storage_sizes(const at::Tensor& self)
```

## 功能说明

获取NPU tensor的内存大小，返回值类型vector<int64_t>，表示获取的NPU tensor内存大小。

## 参数说明

self：Tensor类型，待获取内存大小的tensor。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

