# (beta) torch_npu.contrib.module.PSROIPool

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Performs a position-sensitive region of interest pooling (PSROIPool) operation using the NPU API.

## Prototype

```python
torch_npu.contrib.module.PSROIPool(nn.Module)
```

## Parameters

- **`pooled_height`** (`int`): Pooled output height.
- **`pooled_width`** (`int`): Pooled output width.
- **`spatial_scale`** (`float`): Scale factor applied to the input bounding boxes.
- **`group_size`** (`int`): Number of groups in the position-sensitive score map.
- **`output_dim`** (`int`): Number of output channels.

## Return Values

`Tensor` 

Computation result tensor with shape `(k, 5)` or `(k, 1)`. Labels are zero-based.

## Constraints

Only configurations where `pooled_height == pooled_width == group_size` are supported.

## Example

```python
>>> from torch_npu.contrib.module import PSROIPool
>>> model = PSROIPool(pooled_height=7, pooled_width=7, spatial_scale=1 / 16.0, group_size=7, output_dim=22)
```
