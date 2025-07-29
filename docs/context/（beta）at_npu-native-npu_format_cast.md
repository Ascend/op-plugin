# （beta）at_npu::native::npu_format_cast

## 定义文件

torch_npu\csrc\core\npu\NPUFormat.h

## 函数原型

```
at::Tensor at_npu::native::npu_format_cast(const at::Tensor& self, int64_t acl_format)
```

## 功能说明

NPU tensor格式转换，返回值类型Tensor，表示转换后的tensor。

## 参数说明

self：Tensor类型，待转换格式的tensor。

acl_format：int64_t型，待转换的格式。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

