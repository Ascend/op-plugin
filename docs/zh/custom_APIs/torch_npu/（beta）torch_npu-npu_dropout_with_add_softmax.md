# （beta）torch_npu.npu_dropout_with_add_softmax

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

实现axpy_v2、softmax_v2、drop_out_domask_v3功能，即：

y=x1+ self \*alpha

Softmax(xi)= exp(xi)/∑jexp(xj)

对加权和`y = x1 + alpha * self`做Softmax，得到`result_softmax`。按`mask`将其中被丢弃的元素置为0，保留的元素乘以`1 / (1 - prob)`，得到`result_dropout`。

## 函数原型

```python
torch_npu.npu_dropout_with_add_softmax(self, x1, alpha, prob, dim) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **self**（`Tensor`）：加权加法中乘以`alpha`的输入张量，形状为`(N, C, H, W)`。
- **x1**（`Tensor`）：参与加权加法运算的输入张量，4维张量，shape为(N, C, H, W)。
- **alpha**（`Float`）：缩放系数，支持正负浮点值。
- **prob**（`Float`）：Dropout 丢弃概率，控制 Dropout 操作中元素被随机丢弃的概率，prob=0.0时无元素丢弃（等价于不执行 Dropout），prob越大丢弃元素越多。若prob ≥ 1.0会触发参数非法异常。
- **dim**（`int`）：Softmax 计算的维度。当前仅支持dim=-1。

## 返回值说明

- **mask**（`Tensor`）：一维`torch.uint8`张量，长度为`x1.numel()`向上对齐到128的倍数。掩码作用于`result_softmax`，0表示丢弃，1表示保留。
- **result_softmax** (`Tensor`)：输出张量，shape为(N, C, H, W)。融合计算的最终 Softmax 结果。指定维度dim上的所有元素求和结果为1。
- **result_dropout** (`Tensor`)：输出张量，shape为(N, C, H, W)。对result_softmax执行Dropout后的结果。

## 约束说明

- `self`和`x1`的shape相同。
- H和W是[128, 256, 384, 512]其中之一。
- (N \* C)%32结果为0。
- `dim`为-1。

## 调用示例

输入为两个形状相同的NPU张量。沿末维应用softmax，打印第三个返回值（应用dropout后的结果）的形状。

```python
import torch
import torch_npu

x = torch.rand(16, 16, 128, 128, device="npu")
other = torch.rand_like(x)
mask, softmax_result, output = torch_npu.npu_dropout_with_add_softmax(x, other, 2, 0.9, -1)
print(output.shape)
```

输出：

```text
torch.Size([16, 16, 128, 128])
```
