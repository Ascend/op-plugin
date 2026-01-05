# （beta）torch_npu.npu_transpose

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

返回原始张量视图，其维度已permute，结果连续。支持FakeTensor模式。

## 函数原型

```
torch_npu.npu_transpose(self, perm, require_contiguous=True) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：输入张量。
- **perm** (`List[int]`)：对应维度排列。
- **require_contiguous** (`bool`)：用户是否需要对输入Tensor转为连续，默认值为True。设置为False时，表示不对输入Tensor转为连续。当用户明确输入Tensor为连续Tensor或转置Tensor时，才能设置为True。

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

