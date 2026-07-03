# (beta) torch_npu.contrib.function.fuse_add_softmax_dropout

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

* Description: Fuses the computation logic of add, softmax, and dropout to improve execution performance on the NPU.

* Equivalent computation logic:

    `npu_fuse_add_softmax_dropout` can be used as an equivalent replacement for `torch_npu.contrib.function.fuse_add_softmax_dropout`. The computation logic of the two operators is identical.

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

## Prototype

```python
torch_npu.contrib.function.fuse_add_softmax_dropout(training, dropout, attn_mask, attn_scores, attn_head_size, p=0.5, dim=-1) -> Tensor
```

## Parameters

- **`training`** (`bool`): Required. Specifies whether to use training mode.
- **`dropout`** (`nn.Module`): Required. Dropout layer.
- **`attn_mask`** (`Tensor`): Required. Attention mask.
- **`attn_scores`** (`Tensor`): Required. Original attention scores.
- **`attn_head_size`** (`float`): Required. Head size.
- **`p`** (`float`): Optional. Probability that an element is zeroed. The default value is `0.5`.
- **`dim`** (`int`): Optional. Dimension for softmax computation. The default value is `-1`.

## Return Values

`Tensor`

Final computation result.

## Example

```python
>>> import torch
>>> from torch_npu.contrib.function import fuse_add_softmax_dropout
>>> training = True
>>> dropout = torch.nn.DropoutWithByteMask(0.1)
>>> npu_input1 = torch.rand(96, 12, 30, 30).half().npu()
>>> npu_input2 = torch.rand(96, 12, 30, 30).half().npu()
>>> alpha = 0.125
>>> axis = -1
>>> output = fuse_add_softmax_dropout(training, dropout, npu_input1, npu_input2, alpha, dim=axis)
>>> print(output.shape)
torch.Size([96, 12, 30, 30])
```
