# （beta）at_npu::native::get_npu_format

## 定义文件

torch_npu\csrc\core\npu\NPUFormat.h

## 函数原型

```
int64_t at_npu::native::get_npu_format(const at::Tensor& self)
```

## 功能说明

获取NPU tensor格式信息，返回值类型int64_t，表示获取的NPU tensor格式信息。

>**须知：**<br>
>该接口通常配合申请NPU私有格式内存empty_with_format使用。

## 参数说明

self：Tensor类型，待获取格式信息的tensor。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

