# （beta）at_npu::native::npu_format_cast

## 定义文件

torch_npu\csrc\core\npu\NPUFormat.h

## 函数原型

```cpp
at::Tensor at_npu::native::npu_format_cast(const at::Tensor& self, int64_t acl_format, c10::optional<int64_t> customize_dtype = c10::nullopt)
```

## 功能说明

NPU tensor格式转换，返回值类型Tensor，表示转换后的tensor。

## 参数说明

self：Tensor类型，待转换格式的tensor。

acl_format：int64_t型，待转换的格式。

customize_dtype：可选参数，用于指定格式转换时的目标数据类型。该参数可控制C0值，默认值为`c10::nullopt`。

- 不传参时默认值为`c10::nullopt`，float32和int32数据类型的默认C0值为16，int8数据类型的默认C0值为32；
- 传入`3`（对应int(torch.int32)）时，FRACTAL_NZ格式的C0值为8。

## 约束说明

`customize_dtype`参数仅在Atlas A2 训练系列产品/Atlas A3 训练系列产品且CANN版本为9.1.0及以上的场景下支持。其他产品或CANN 9.1.0以下版本，传入该参数将导致异常。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>
