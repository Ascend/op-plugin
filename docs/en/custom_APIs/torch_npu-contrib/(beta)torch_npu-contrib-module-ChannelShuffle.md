# (beta) torch_npu.contrib.module.ChannelShuffle

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

- Description: Applies an NPU-optimized channel shuffle operation.

- Equivalent computation logic:
  
    In the `split_shuffle=False` scenario, `cpu_channel_shuffle` can be used as an equivalent replacement for `torch_npu.contrib.module.ChannelShuffle`. The computation logic of the two operators is identical.

  ```python
  import torch
  def cpu_channel_shuffle(x, groups, split_shuffle):
      # CPU supports only `split_shuffle=False`
      batchsize, num_channels, height, width = x.size()
      channels_per_group = num_channels // groups
      x.requires_grad_(True)
      # reshape
      x = x.view(batchsize, groups, channels_per_group, height, width)
    
      x = torch.transpose(x, 1, 2).contiguous()
    
      # flatten
      x = x.view(batchsize, -1, height, width)
      output = x.view(batchsize, -1, height, width)
      return output
  ```
  
## Prototype

```python
torch_npu.contrib.module.ChannelShuffle(in_channels, groups=2, split_shuffle=True)
```

## Parameters

**Computation Parameters**

- **`in_channels`** (`int`): Required. Total number of channels in the input tensor.
- **`groups`** (`int`): Optional. Number of shuffle groups. The default value is `2`.
- **`split_shuffle`** (`bool`): Optional. Specifies whether to perform a `chunk` operation after channel shuffling. The default value is `True`.

**Computation Input**

- **`x1`** (`Tensor`): Input tensor. The shape is `(N, C_in, *)`.
- **`x2`** (`Tensor`): Input tensor. The shape is `(N, C_in, *)`.

## Return Values

- **`out1`** (`Tensor`): Output tensor with shape `(N, C_out, *)`.
- **`out2`** (`Tensor`): Output tensor with shape `(N, C_out, *)`.

## Constraints

Only scenarios where `groups` is set to `2` are implemented. Modify the implementation manually for other values of `groups`.

## Example

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.module import ChannelShuffle
>>> x1 = torch.randn(2, 32, 7, 7).npu()
>>> x2 = torch.randn(2, 32, 7, 7).npu()
>>> m = ChannelShuffle(64, split_shuffle=True)
>>> out1, out2 = m(x1, x2)
>>> print(out1.shape)
torch.Size([2, 32, 7, 7])
>>> print(out2.shape)
torch.Size([2, 32, 7, 7])
```
