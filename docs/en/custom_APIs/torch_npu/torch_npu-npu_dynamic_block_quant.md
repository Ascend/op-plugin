# torch_npu.npu_dynamic_block_quant

## Supported Products

| Product                                                     | Supported|
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 training products</term>                       |    √     |
| <term>Atlas A3 inference products</term>                       |    √     |
| <term>Atlas A2 training products</term>                       |    √     |
| <term>Atlas A2 inference products</term>|    √     |

## Function

- Description: Divides the input tensor into multiple data blocks based on the specified `row_block_size` and `col_block_size`, and performs quantization at the block level. For each block, the corresponding quantization parameter `scale` is first computed, and the input is quantized using this `scale`. The final quantized result and the per-block `scale` parameters are returned.

- Formulas:
  $$
  input\_max = block\_reduce\_max(abs(x))
  $$

  $$
  scale = min(FP8\_MAX(HiF8\_MAX / INT8\_MAX) / input\_max, 1/min\_scale)
  $$

  $$
  y = cast\_to\_[FP8/HiF8/INT8](x / scale)
  $$

  $block\_reduce\_max$ represents the maximum value within each individual block.

## Prototype

```python
torch_npu.npu_dynamic_block_quant(x, *, min_scale=0.0, round_mode="rint", dst_type=1, row_block_size=1, col_block_size=128) -> (Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Input tensor to be quantized. The data type can be `float16` or `bfloat16`. Non-contiguous tensors are supported. The data layout can be ND. The shape of this parameter must have two or three dimensions.
- **`min_scale`** (`float`): Optional. Minimum scale threshold value participating in the `scale` computation. The value must be greater than or equal to 0.
- **`round_mode`** (`str`): Optional. Rounding conversion mode used when casting to the output format. Currently, only `rint` is supported.
- **`dst_type`** (`int`): Optional. Target data type of the output tensor `y`. Currently, only the value `1` is supported, indicating that the data type of the output `y` is `int8`.
- **`row_block_size`** (`int`): Optional. Row size of a single quantization data block. Currently, only `1` is supported.
- **`col_block_size`** (`int`): Optional. Column size of a single quantization data block. Currently, only `128` is supported.

## Return Values

- **`y`** (`Tensor`): Quantization result.
- **`scale`** (`Tensor`): Scale parameter used during quantization.

## Example

  ```python
  >>> import torch
  >>> import torch_npu
  
  >>> x = torch.rand(3, 4).to("npu").to(torch.float16)
  >>> min_scale = 0
  >>> dst_type = 1
  >>> row_block_size = 1
  >>> col_block_size = 128
  
  >>> y, scale = torch_npu.npu_dynamic_block_quant(x, min_scale=min_scale, dst_type=dst_type, row_block_size=row_block_size, col_block_size=col_block_size)
  >>> y
  tensor([[ 92,  65,  15, 127],
          [100, 127, 116,  64],
          [ 95,  15,  87, 127]], device='npu:0', dtype=torch.int8)
  >>> scale
  tensor([[0.0063],
          [0.0076],
          [0.0073]], device='npu:0')
  ```
