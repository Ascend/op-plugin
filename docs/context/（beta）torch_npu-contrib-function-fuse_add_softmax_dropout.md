# （beta）torch_npu.contrib.function.fuse_add_softmax_dropout

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

使用NPU自定义算子替换原生写法，以提高性能。

## 函数原型

```
torch_npu.contrib.function.fuse_add_softmax_dropout(training, dropout, attn_mask, attn_scores, attn_head_size, p=0.5, dim=-1)
```

## 参数说明

- **training** (`bool`)：是否为训练模式。
- **dropout** (`nn.Module`): dropout层。
- **attn_mask** (`Tensor`)：注意力掩码。
- **attn_scores** (`Tensor`)：原始attention分数。
- **attn_head_size** (`float`)：head size。
- **p** (`float`)：元素被归零的概率，默认值为0.5。
- **dim** (`int`)：待计算softmax的维度，默认值为-1。

## 返回值说明
`Tensor`

mask操作的结果。

## 调用示例

```python
>>> from torch_npu.contrib.function import fuse_add_softmax_dropout
>>> training = True
>>> dropout = torch.nn.DropoutWithByteMask(0.1)
>>> npu_input1 = torch.rand(96, 12, 30, 30).half().npu()
>>> npu_input2 = torch.rand(96, 12, 30, 30).half().npu()
>>> alpha = 0.125
>>> axis = -1
>>> output = fuse_add_softmax_dropout(training, dropout, npu_input1, npu_input2, alpha, p=axis)
```

