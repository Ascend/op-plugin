# torch_npu.npu_dynamic_block_quant

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->

## Function

- Function: divides the input tensor into multiple data blocks according to the given `row_block_size` and `col_block_size`, and performs quantization at the granularity of data blocks. In each block, the quantization parameter `scale` corresponding to the current block is computed first, and the input is quantized according to `scale`. The final quantization result and the quantization parameter `scale` of each block are output.

- Formulas:
  $$
  input\_max = block\_reduce\_max(abs(x))
  $$

  $$
  scale =
  \begin{cases}
  min(input\_max / FP8\_MAX, 1 / min\_scale), & dst\_type \text{ is specified as FP8} \\
  min(input\_max / HiF8\_MAX, 1 / min\_scale), & dst\_type \text{ is specified as HiF8} \\
  min(input\_max / INT8\_MAX, 1 / min\_scale), & dst\_type \text{ is specified as INT8}
  \end{cases}
  $$

  $$
  y = cast\_to\_[FP8/HiF8/INT8](x / scale)
  $$

  In the formula, $block\_reduce\_max$ indicates the maximum value in each `block`. `FP8_MAX`, `HiF8_MAX`, and `INT8_MAX` respectively indicate the maximum positive values representable by the FP8, HiF8, and INT8 target quantization types, which are determined by `dst_type`. When `dst_type_max` is not 0, the value of `dst_type_max` is used as the maximum value of the target type.
  FP8, HiF8, and INT8 are all 8-bit low-precision quantization target types, respectively indicating the FP8 floating-point, HiF8 floating-point, and INT8 integer formats. The currently supported maximum values are as follows:

  | Target Type | Maximum Positive Value |
  | --- | --- |
  | FP8 | FP8_MAX = 448 |
  | HiF8 | HiF8_MAX = 32768 |
  | INT8 | INT8_MAX = 127 |

## Prototype

```python
torch_npu.npu_dynamic_block_quant(x, *, min_scale=0.0, round_mode="rint", dst_type=1, row_block_size=1, col_block_size=128, dst_type_max=0.0) -> (Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Input tensor to be quantized. The data type can be `torch.float16` or `torch.bfloat16`. Non-contiguous tensors are supported. The data layout can be ND. The shape of this parameter must have two or three dimensions. Empty tensors are not supported.
- **`min_scale`** (`float`): Optional. Minimum scale threshold value participating in the `scale` computation. The value must be greater than or equal to 0.
- **`round_mode`** (`str`): Optional. Rounding conversion mode used when casting to the output format. Currently, only `rint` is supported.

    <!-- npu="A3,910b" id4 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: only `"rint"` is supported.
    <!-- end id4 -->
    <!-- npu="950" id5 -->
    - <term>Ascend 950PR&950DT products</term>:
      - When `dst_type` is `torch.int8`, `torch.float8_e4m3fn`, or `torch.float8_e5m2`, only `"rint"` is supported.
      - When `dst_type` is `torch_npu.hifloat8`, only `"round"` is supported.
    <!-- end id5 -->

- **`dst_type`** (`int`): Optional. Target data type of the output tensor `y`. Currently, only the value `1` is supported, indicating that the data type of the output `y` is `torch.int8`.

    <!-- npu="A3,910b" id6 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the data type can be `torch.int8`.
    <!-- end id6 -->
    <!-- npu="950" id7 -->
    - <term>Ascend 950PR&950DT products</term>: the data type can be `torch.int8`, `torch_npu.hifloat8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`.
    <!-- end id7 -->

- **`row_block_size`** (`int`): Optional. Row size of a single quantization data block. Currently, only `1` is supported.

    <!-- npu="A3,910b" id8 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: only 1 is supported.
    <!-- end id8 -->
    <!-- npu="950" id9 -->
    - <term>Ascend 950PR&950DT products</term>: 1, 128, 256, and 512 are supported.
    <!-- end id9 -->

- **`col_block_size`** (`int`): Optional. Column size of a single quantization data block. Currently, only `128` is supported.

    <!-- npu="A3,910b" id10 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: only 1 or 128 is supported.
    <!-- end id10 -->
    <!-- npu="950" id11 -->
    - <term>Ascend 950PR&950DT products</term>: 64, 128, 192, and 256 are supported.
    <!-- end id11 -->

- **`dst_type_max`** (`float`): Optional. Maximum value of the target data type. The default value is 0.0. When the value is 0.0, the original maximum value of the data type is used.

    <!-- npu="A3,910b" id12 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: this parameter is not supported currently.
    <!-- end id12 -->
    <!-- npu="950" id13 -->
    - <term>Ascend 950PR&950DT products</term>: it takes effect only when `dst_type` is `torch_npu.hifloat8`, and the supported values are 0.0, 15.0, 56.0, 224.0, and 32768.0.
    <!-- end id13 -->

## Return Values

- **`y`** (`Tensor`): Quantization result. The data type is specified by the `dst_type` parameter, and the shape is the same as that of the input `x`.

    <!-- npu="950" id14 -->
    - <term>Ascend 950PR&950DT products</term>:
        - Single-operator mode/static graph mode: when `dst_type` is `torch_npu.hifloat8`, the output data type of `y` is `torch.uint8` (which actually carries the `torch_npu.hifloat8` type).
        - Dynamic graph mode: when `dst_type` is `torch_npu.hifloat8`, the output data type of `y` is `torch.bits8` (which actually carries the `torch_npu.hifloat8` type).
    <!-- end id14 -->

- **`scale`** (`Tensor`): Quantization parameter used during quantization. The data type is `torch.float32`. If the shape of the input `x` is `[M, N]`, the shape of `scale` is `[ceil(M/row_block_size), ceil(N/col_block_size)]`. If the shape of the input `x` is `[B, M, N]`, the shape of `scale` is `[B, ceil(M/row_block_size), ceil(N/col_block_size)]`.

## Constraints

- This API can be used in inference and training scenarios.
- This API supports single-operator mode and TorchAir graph mode.

## Example

- Single-operator call

    ```python
  >>> import torch
  >>> import torch_npu
  
  >>> x = torch.rand(3, 4).to("npu").to(torch.float16)
  >>> min_scale = 0
  >>> dst_type = 1
  >>> row_block_size = 1
  >>> col_block_size = 128
  
  >>> y, scale = torch_npu.npu_dynamic_block_quant(x, min_scale=min_scale, dst_type=dst_type, row_block_size=row_block_size, col_block_size=col_block_size)
  >>> print(y)
  tensor([[ 92,  65,  15, 127],
          [100, 127, 116,  64],
          [ 95,  15,  87, 127]], device='npu:0', dtype=torch.int8)
  >>> print(scale)
  tensor([[0.0063],
          [0.0076],
          [0.0073]], device='npu:0')
  ```

<!-- npu="950" id15 -->
- Graph mode call: Only <term>Ascend 950PR&950DT products</term> are supported.

    ```python
    import torch
    import torch_npu
    import torchair

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, min_scale=0.0, round_mode="rint", dst_type=torch.float8_e5m2, row_block_size=1, col_block_size=128):
            return torch_npu.npu_dynamic_block_quant(x, min_scale=min_scale, round_mode=round_mode, dst_type=dst_type, row_block_size=row_block_size, col_block_size=col_block_size)

    x = torch.zeros((1, 128)).to(torch.float16).to('npu')
    model = Model()
    model = model.to('npu')
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=True)
    y, scale = model(x, 0.0, "rint", torch.float8_e5m2, 1, 128)
    ```
<!-- end id15 -->
