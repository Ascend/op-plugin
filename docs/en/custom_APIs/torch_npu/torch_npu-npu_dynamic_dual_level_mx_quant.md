# torch\_npu.npu\_dynamic\_dual\_level\_mx\_quant

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product | Supported |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## Function

- Implements MX quantization with `FLOAT4`-class data types as the target data type. Only the last axis of the input tensor is quantized, while all other axes are treated as a single combined axis.
- Formulas:
    1. Input tensor `x` is divided into groups of `k_0 = 512` elements along the last axis. Each group contains `k_0` elements, represented as $x_{i=1}^{k_0}$, and each data block undergoes first-level dynamic quantization. The quantization scale and first-level quantization result for each group are calculated as follows. These individual results are then combined to obtain the quantization scale `level0_scale` and the first-level quantization result `temp`.

        $$
        input\_max_i = max_i(abs(x_i))
        $$

        $$
        level0\_scale_i = input\_max_i / (FP4\_E2M1\_MAX)
        $$

        $$
        temp_i = cast\_to\_x\_type(x_i / level0\_scale_i), \space i \space \text{ranges from} \space 1 \space \text{to} \space 512
        $$

    2. The resulting `temp` is then divided into groups of `k_1 = 32` elements along the last axis. Each group contains `k_1` elements, represented as $temp_{i=1}^{k_1}$, and each data block undergoes second-level dynamic quantization. The quantization scale for each group is calculated as follows. These individual results are then combined to obtain the quantization scale `level1_scale`.

        $$
        shared\_exp_i = floor(log_2(max_i(|temp_i|))) - emax
        $$

        $$
        level1\_scale_i = 2^{shared\_exp_i}
        $$

    3. Finally, the data type is converted according to `round_mode` to obtain the quantized result $y_i$ for each group.

        $$
        y_i = cast\_to\_FP4\_E2M1(temp_i / level1\_scale_i, round_mode), \space i \space \text{ranges from} \space 1 \space \text{to} \space 32
        $$

        Quantized $y_i$ values are arranged according to the positions of the corresponding $x_i$ elements to form the output $y$. Output `level0_scale` is formed by grouping $level0\_scale_i$ values along the last axis, and output `level1_scale` is formed by grouping $level1\_scale_i$ values along the last axis.

        $max_i$ represents the maximum value in the $i$-th group, and `emax` represents the exponent of the maximum finite normal positive number of the corresponding data type. The mappings are defined as follows:

        | dst_type | emax |
        | --- | --- |
        | float4_e2m1fn_x2 | 2 |

## Prototype

```python
torch_npu.npu_dynamic_dual_level_mx_quant(input, *, smooth_scale=None, round_mode="rint") -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`input`** (`Tensor`): Required. Data to be quantized, $x_i$ in the formulas. This parameter must be 1D to 7D, and the last dimension must be even. Non-contiguous tensors are supported. The data layout can be `ND`. The data type can be `bfloat16` or `float16`. Empty tensors are not supported.
- **`*`**: Position delimiter indicating preceding variables are position-dependent arguments that must be entered in order, while succeeding variables are optional keyword arguments that must be assigned using key-value pairs in any order (omitted arguments will use default values).
- **`smooth_scale`** (`Tensor`): Optional. Functionality currently not supported (default value can be passed).
- **`round_mode`** (`str`): Optional. Data conversion mode, corresponding to $round\_mode$ in the formulas. Supported values are `"rint"`, `"round"`, and `"floor"` (default value: `"rint"`).

## Return Values

- **`y`** (`Tensor`): Quantization result, $y_i$ in the formulas. The logical data type can be `float4_e2m1fn_x2`, but the actual returned data type is `uint8`. The size of the last dimension is half that of the `input` (manual unpacking required to retrieve the actual values). The data layout can be `ND`.
- **`level0_scale`** (`Tensor`): Scale for first-level quantization, $level0\_scale_i$ in the formulas. The data type can be `float32`. The size of the last dimension equals the size of the last dimension of `input` divided by 512 and rounded up. The data layout can be `ND`.

- **`level1_scale`** (`Tensor`): Scale for second-level quantization, $level1\_scale_i$ in the formulas. The logical data type can be `float8_e8m0fnu`, but the actual returned data type is `uint8` (manual conversion required to retrieve the actual values). The number of dimensions is equal to the number of dimensions of `input` plus 1. The sizes of the last two dimensions are `[((ceil(input.shape[-1] / 32) + 2 - 1) / 2), 2]`, with even padding applied using `0` as the padding value. The data layout can be `ND`.

## Constraints

- API supported in training and inference scenarios.
- API supports single-operator mode and graph mode calls.
- Shape constraints between `input` and outputs `y`, `level0_scale`, and `level1_scale`:
  - rank\(level1\_scale\) = rank\(input\) + 1
  - level0\_scale.shape\[-1\] = ceil\(input.shape\[-1\] / 512\)
  - level1\_scale.shape\[-2\] = \(ceil\(input.shape\[-1\] / 32\) + 2 - 1\) / 2
  - level1\_scale.shape\[-1\] = 2
  - Other dimensions are identical to those of `input`.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    
    input = torch.randn((1, 512), dtype=torch.bfloat16).npu()
    y_tmp, level0_scale_tmp, level1_scale_tmp = torch_npu.npu_dynamic_dual_level_mx_quant(
        input,
        smooth_scale=None,
        round_mode="rint")
    y = y_tmp.cpu()
    level0_scale = level0_scale_tmp.cpu()
    level1_scale = level1_scale_tmp.cpu().view(torch.float8_e8m0fnu)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair
    
    class DynamicDualLevelMxQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, smooth_scale=None, round_mode='rint'):
            return torch_npu.npu_dynamic_dual_level_mx_quant(x, smooth_scale=smooth_scale, round_mode=round_mode)
    def dynamic_dual_level_mx_quant_test():
        input = torch.randn((1, 512), dtype=torch.bfloat16).npu()
        model = DynamicDualLevelMxQuantModel()
        model.to('npu')
    
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)
    
        y_tmp, level0_scale_tmp, level1_scale_tmp = model(input, smooth_scale=None)
    
        y = y_tmp.cpu()
        level0_scale = level0_scale_tmp.cpu()
        level1_scale = level1_scale_tmp.cpu()
    if __name__ == "__main__":
        dynamic_dual_level_mx_quant_test()
    ```
