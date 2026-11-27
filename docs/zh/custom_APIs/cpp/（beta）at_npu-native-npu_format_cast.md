# （beta）at_npu::native::npu_format_cast

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id5 -->

## 功能说明

NPU Tensor格式转换，返回值类型Tensor，表示转换后的Tensor。

## 定义文件

torch_npu/csrc/core/npu/NPUFormat.h

## 函数原型

```cpp
at::Tensor at_npu::native::npu_format_cast(const at::Tensor& self, int64_t acl_format);
```

## 参数说明

self：Tensor类型，待转换格式的Tensor。

**acl_format** (`int64_t`)：必选参数，取值为ACL格式枚举。例如，ND为`ACL_FORMAT_ND`（`2`），FRACTAL_NZ为`ACL_FORMAT_FRACTAL_NZ`（`29`）。可用的格式组合取决于芯片和输入Tensor。

如需指定目标布局的数据类型，可参见 [Python npu_format_cast](../torch_npu/（beta）torch_npu-npu_format_cast.md) 的`customize_dtype`参数。

## 约束说明

<!-- npu="950" id6 -->
<term>Ascend 950DT系列产品</term>场景下，本接口转为私有格式（如FRACTAL_NZ）的行为不受`allow_internal_format = False`约束。
<!-- end id6 -->

<!-- npu="950" id7 -->
<term>Ascend 950DT系列产品</term>场景下，将Tensor转为FRACTAL_NZ格式时，当前不支持以下特殊场景：

- 当`self`的数据类型为`torch.float16`、`torch.bfloat16`时，若`self`维度表示为[k, n]，则k为1场景暂不支持。
- 调用本接口转为FRACTAL_NZ格式后，不支持进行任何能修改Tensor的操作，包括contiguous、pad、view、slice等。
- `self`的shape后两维任意一维度shape等于1场景，不允许转FRACTAL_NZ后进行transpose。
<!-- end id7 -->
