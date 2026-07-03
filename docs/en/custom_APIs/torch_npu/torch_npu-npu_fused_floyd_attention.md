# torch_npu.npu_fused_floyd_attention

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/release_notes/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E).

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 training products/Atlas A2 inference products</term>           |    √     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>              | √   |

## Function

- Description: In training scenarios, `npu_fused_floyd_attention` differs from traditional FlashAttention (`npu_fusion_attention`) by treating the sequence dimension (`seq`) as an additional batch axis during QK/PV attention computation, thereby converting the attention computation into batch matrix multiplication (`batchMatmul`).
- Formula:

$$
\text{weights} = \text{Softmax}\left(\text{atten\_mask} + \text{scale\_value} \cdot \left(\text{einsum}(\text{query}, \text{key}_1^T) + \text{einsum}(\text{query}, \text{key}_2^T)\right)\right)
$$

$$
\text{attention\_out} = \text{einsum}(\text{weights}, \text{value}_1) + \text{einsum}(\text{weights}, \text{value}_2)
$$

## Prototype

```python
torch_npu.npu_fused_floyd_attention(query_ik, key_ij, value_ij, key_jk, value_jk, *, atten_mask=None, scale_value=1.) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`query_ik`** (`Tensor`): Required. Input tensor, corresponding to `query` in the formula. The data type can be `bfloat16` or `float16`. The data layout can be ND. The input shape can be [BHNMD].
- **`key_ij`** (`Tensor`): Required. Input tensor, $key_1$ in the formula, representing the relationship or feature from node i to its direct neighbor j. The data type can be `bfloat16` or `float16`. The data layout can be ND. The input shape can be [BHNKD].
- **`value_ij`** (`Tensor`): Required. Input tensor, $value_1$ in the formula, representing the information content from node i to its direct neighbor j. The data type can be `bfloat16` or `float16`. The data layout can be ND. The input shape can be [BHNKD].
- **`key_jk`** (`Tensor`): Required. Input tensor, $key_2$ in the formula, representing the relationship or feature from the direct neighbor j to pivot k. The data type can be `bfloat16` or `float16`. The data layout can be ND. The input shape can be [BHKMD].
- **value_jk** (`Tensor`): (required) input tensor, $value_2$ in the formula, indicating the information from node j to its direct neighbor k. The data type can be `bfloat16`, `float16` or `float32`. The data layout can be ND. The input shape can be [BHKMD].
- **`atten_mask`** (`Tensor`): Optional. Input tensor, $atten\_mask$ in the formula. The data type can be `bool` or `uint8`. The data layout can be ND. The input shape must be [B1N1K]. A value of `1` indicates that the bit is excluded from computation. A value of `0` indicates that the bit is included in computation. The default value is `None`.
- **`scale_value`** (`float`): Optional. Scaling factor, $scale\_value$ in the formula. The data type can be `float`. The default value is `1`.

## Return Values

- **`softmax_max_out`** (`Tensor`): Output tensor, intermediate Max result of Softmax computation, used for backward computation. The data type can be `float`. The output shape is [BHNM8]. The data layout can be ND.
- **`softmax_sum_out`** (`Tensor`): Output tensor, intermediate Sum result of Softmax computation, used for backward computation. The data type can be `float`. The output shape is [BHNM8]. The data layout can be ND.
- **`attention_out`** (`Tensor`): Output tensor, final computation output, $attention\_out$ in the formula. The data type can be `bfloat16` or `float16`. The data type and shape must match those of `query_ik`. The data layout can be ND. The input shape can be `[BHNMD]`.

## Constraints

- Shape constraints:

    `B`: The value ranges from 1 to 2K.

    `H`: The value ranges from 1 to 256.

    `N`: The value ranges from 16 to 1M and must be a multiple of 16.

    `M`: The value ranges from 128 to 1M and must be a multiple of 128.

    `K`: The value ranges from 128 to 1M and must be a multiple of 128.

    `D`: Only `32`, `64`, and `128` are supported.

- Axes 0, 2, and 4 of `query_ik` and `key_ij` must be identical.
- The shapes of `key_ij` and `value_ij` must be identical.
- The shapes of `key_jk` and `value_jk` must be identical.
- The shapes of `softmax_max_out` and `softmax_sum_out` must be identical.
- The data types of `query_ik`, `key_ij`, `value_ij`, `key_jk`, and `value_jk` must be identical.
- PyTorch 2.6.0 or later is supported.
- Backward computation for this API does not support determinism.

## Examples

```python
import torch
import torch_npu
import math

def truncated_normal(mean, std, min, max, size):
    x = torch.normal(mean, std, size)
    x = torch.where((x < min) | (x > max), torch.tensor(0.0), x)
    return x
B, N, S1, S2, S3, D = 1, 1, 16, 256, 256, 64
Q = truncated_normal(0.0, 1, -10, 10, (B, N, S1, S2, D)).to(torch.bfloat16).npu()
K1 = truncated_normal(0.0, 1, -10, 10, (B, N, S1, S3, D)).to(torch.bfloat16).npu()
K2 = truncated_normal(0.0, 1, -10, 10, (B, N, S3, S2, D)).to(torch.bfloat16).npu()
V1 = truncated_normal(0.0, 1, -10, 10, (B, N, S1, S3, D)).to(torch.bfloat16).npu()
V2 = truncated_normal(0.0, 1, -10, 10, (B, N, S3, S2, D)).to(torch.bfloat16).npu()
atten_mask = torch.randint(0, 2, [B, 1, S1, 1, S3]).to(torch.bool).npu()
scale = 1.0/math.sqrt(D)
x_max_npu, x_sum_npu, output_npu = torch_npu.npu_fused_floyd_attention(
    Q,
    K1,
    V1,
    K2,
    V2,
    atten_mask = atten_mask,
    scale_value = scale
)
```
