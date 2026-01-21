# （beta）torch_npu.contrib.function.fuse_add_softmax_dropout

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

* API功能：融合了add、softmax和dropout的计算逻辑，以提高在NPU上计算的性能。

* 等价计算逻辑：

    可使用`npu_fuse_add_softmax_dropout`等价替换`torch_npu.contrib.function.fuse_add_softmax_dropout`，两者计算逻辑一致。
    ```python
    import torch
    import math
    import torch.nn.functional as F
    def npu_fuse_add_softmax_dropout(dropout, attn_mask, attn_scores, attn_head_size):
        attn_scores = torch.add(attn_mask, attn_scores, alpha=(1 / math.sqrt(attn_head_size)))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = dropout(attn_probs)
        return attn_probs
    ```
## 函数原型

```
torch_npu.contrib.function.fuse_add_softmax_dropout(training, dropout, attn_mask, attn_scores, attn_head_size, p=0.5, dim=-1) -> Tensor
```

## 参数说明

- **training** (`bool`)：必选参数，是否为训练模式。
- **dropout** (`nn.Module`): 必选参数，dropout层。
- **attn_mask** (`Tensor`)：必选参数，注意力掩码。
- **attn_scores** (`Tensor`)：必选参数，原始attention分数。
- **attn_head_size** (`float`)：必选参数，head size。
- **p** (`float`)：可选参数，元素被归零的概率，默认值为0.5。
- **dim** (`int`)：可选参数，待计算softmax的维度，默认值为-1。

## 返回值说明
`Tensor`

返回计算结果。

## 调用示例

```python
>>> import torch
>>> from torch_npu.contrib.function import fuse_add_softmax_dropout
>>> training = True
>>> dropout = torch.nn.DropoutWithByteMask(0.1)
>>> npu_input1 = torch.rand(96, 12, 30, 30).half().npu()
>>> npu_input2 = torch.rand(96, 12, 30, 30).half().npu()
>>> alpha = 0.125
>>> axis = -1
>>> output = fuse_add_softmax_dropout(training, dropout, npu_input1, npu_input2, alpha, p=axis)
>>> output.shape
torch.Size([96, 12, 30, 30])
```
