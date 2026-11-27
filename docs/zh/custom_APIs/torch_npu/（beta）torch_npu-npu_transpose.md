# （beta）torch_npu.npu_transpose

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，建议使用torch.permute。

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

返回维度已permute的张量，结果连续。支持FakeTensor模式。

## 函数原型

```python
torch_npu.npu_transpose(self, perm, require_contiguous=True) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：输入张量。
- **perm** (`List[int]`)：必选参数，维度的排列。长度等于输入张量的秩，每个维度索引恰好出现一次。
- **require_contiguous** (`bool`)：可选参数，默认值为`True`，由算子按需将输入转为连续张量。确认输入是连续张量或转置视图时，可设置为`False`，跳过输入连续化。

## 调用示例

将形状为 (2, 3, 5) 的张量按 (2, 0, 1) 重排，得到形状 (5, 2, 3)。

```python
import torch
import torch_npu

x = torch.randn(2, 3, 5).npu()
print(x.shape)
x1 = torch_npu.npu_transpose(x, (2, 0, 1))
print(x1.shape)
```

一次运行的输出：

```text
torch.Size([2, 3, 5])
torch.Size([5, 2, 3])
```
