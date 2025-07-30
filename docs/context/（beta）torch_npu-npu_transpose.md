# （beta）torch_npu.npu_transpose

## 函数原型

```
torch_npu.npu_transpose(self, perm, require_contiguous=True) -> Tensor
```

## 功能说明

返回原始张量视图，其维度已permute，结果连续。支持FakeTensor模式。

## 参数说明

- self (Tensor) - 输入张量。
- perm (ListInt) - 对应维度排列。
- require_contiguous(Bool，默认值为True) - 用户是否需要对输入Tensor做转连续。设置为False时，表示不对输入Tensor做转连续。用户明确输入Tensor为连续Tensor或转置Tensor时，才能设置为True。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> x = torch.randn(2, 3, 5).npu()
>>> x.shape
torch.Size([2, 3, 5])
>>> x1 = torch_npu.npu_transpose(x, (2, 0, 1))
>>> x1.shape
torch.Size([5, 2, 3])
>>> x2 = torch_npu.npu_transpose(x, (2, 0, 1))
>>> x2.shape
torch.Size([5, 2, 3])
```

