# （beta）at_npu::native::get_npu_format

## 定义文件

torch_npu\csrc\core\npu\NPUFormat.h

## 函数原型

```cpp
int64_t at_npu::native::get_npu_format(const at::Tensor& self)
```

## 功能说明

获取NPU tensor的格式信息，返回值为int64_t类型。

> [!NOTICE]  
> 该接口通常配合申请NPU私有格式内存empty_with_format使用。

## 参数说明

self：Tensor类型，待获取格式信息的tensor。

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
