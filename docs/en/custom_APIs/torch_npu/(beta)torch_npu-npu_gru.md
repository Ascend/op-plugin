# (beta) torch_npu.npu_gru

> [!NOTICE]  
> This API is planned for deprecation. Use `torch.gru` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Computes DynamicGRUV2.

## Prototype

```python
torch_npu.npu_gru(input, hx, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, has_biases, num_layers, dropout, train, bidirectional, batch_first) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## Parameters

- **`input`** (`Tensor`): The data type can be `float16`. The data layout can be `FRACTAL_NZ`.
- **`hx`** (`Tensor`): The data type can be `float16` or `float32`. The data layout can be `FRACTAL_NZ`.
- **`weight_input`** (`Tensor`): The data type can be `float16`. The data layout can be `FRACTAL_Z`.
- **`weight_hidden`** (`Tensor`): The data type can be `float16`. The data layout can be `FRACTAL_Z`.
- **`bias_input`** (`Tensor`): The data type can be `float16` or `float32`. The data layout can be ND.
- **`bias_hidden`** (`Tensor`): The data type can be `float16` or `float32`. The data layout can be ND.
- **`seq_length`** (`Tensor`): The data type can be `int32`. The data layout can be ND.
- **`has_biases`** (`bool`): The default value is `True`.
- **`num_layers`** (`int`): Number of layers.
- **`dropout`** (`float`): Dropout discard probability.
- **`train`** (`bool`): Specifies whether training is executed inside the operator. The default value is `True`.
- **`bidirectional`** (`bool`): The default value is `True`.
- **`batch_first`** (`bool`): The default value is `True`.

## Return Values

- **`y`** (`Tensor`): The data type can be `float16` or `float32`. The data layout can be `FRACTAL_NZ`.
- **`output_h`** (`Tensor`): The data type can be `float16` or `float32`. The data layout can be `FRACTAL_NZ`.
- **`update`** (`Tensor`): The data type can be `float16` or `float32`. The data layout can be `FRACTAL_NZ`.
- **`reset`** (`Tensor`): The data type can be `float16` or `float32`. The data layout can be `FRACTAL_NZ`.
- **`new`** (`Tensor`): The data type can be `float16` or `float32`. The data layout can be `FRACTAL_NZ`.
- **`hidden_new`** (`Tensor`): The data type can be `float16` or `float32`. The data layout can be `FRACTAL_NZ`.

## Constraints

Currently, this API is not supported when `jit_compile=False`. To use this API under this mode, add `"DynamicGRUV2"` to the `"NPU_FUZZY_COMPILE_BLACKLIST"` option. For details, see [Example of Adding a Binary Blocklist](../blacklist.md).
