# （beta）at_npu::native::npu_dropout_gen_mask

## 定义文件

third_party\op-plugin\op_plugin\include\ops.h

## 函数原型

```
at::Tensor npu_dropout_gen_mask(const at::Tensor &self, at::IntArrayRef size, double p, int64_t seed, int64_t offset, c10::optional<bool> parallel, c10::optional<bool> sync)
```

## 功能说明

训练过程中，按照概率p随机生成mask，用于元素置零。

## 参数说明

- self ：Tensor类型，输入的张量。
- size：IntArrayRef类型，获取tensor的size。
- p：double类型，元素置0的概率。
- seed ：int64_t类型，随机数的种子，影响生成的随机数序列。
- offset ：int64_t类型，随机数的偏移量，影响生成随机数序列的位置。
- parallel ：可选参数bool类型，是否并行计算。
- sync：可选参数bool类型，是否做同步。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

