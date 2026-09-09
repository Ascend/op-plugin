# torch_npu.npu_apply_rotary_pos_emb

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950DT</term>                             |    √    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas inference products</term>                       |    √    |

## Function

Fuses the `query` and `key` operators into a single path to improve inference performance and performs in-place updates during rotary positional embedding computation.

## Prototype

```python
torch_npu.npu_apply_rotary_pos_emb(query, key,  cos, sin, *, layout='BSND', rotary_mode='half') -> (Tensor, Tensor)
```

## Parameters

- **`query`** (`Tensor`): Required. First tensor for rotary position embedding. The data type can be `float32`, `float16`, or `bfloat16`. The data layout can be ND. The shape has 3 dimensions when `layout` is `TND`, and 4 dimensions in other `layout` scenarios.
  - Atlas inference products, Atlas A2 training products/Atlas A2 inference products, and Atlas A3 training products/Atlas A3 inference products: Empty tensors are not supported. The last dimension (`D`) of `shape` must be `128` or `64`.
  - Ascend 950DT: Empty tensors are supported. The last dimension (`D`) of `shape` must be less than or equal to `1024`.
- **`key`** (`Tensor`): Required. Second tensor for rotary position embedding. The data type can be `float32`, `float16`, or `bfloat16`. The data layout can be ND. The shape has 3 dimensions when `layout` is `TND`, and 4 dimensions in other `layout` scenarios.
  - Atlas inference products, Atlas A2 training products/Atlas A2 inference products, and Atlas A3 training products/Atlas A3 inference products: Empty tensors are not supported. The last dimension (`D`) of `shape` must be `128` or `64`.
  - Ascend 950DT: Empty tensors are supported. The last dimension (`D`) of `shape` must be less than or equal to `1024`.
- **`cos`** (`Tensor`): Required. Cosine value tensor for rotary position embedding. The data type can be `float32`, `float16`, or `bfloat16`. The data layout can be ND. The shape has 3 dimensions when `layout` is `TND`, and 4 dimensions in other `layout` scenarios.
  - Atlas inference products, Atlas A2 training products/Atlas A2 inference products, and Atlas A3 training products/Atlas A3 inference products: Empty tensors are not supported. The B dimension in the shape must match that of `query` and `key`. The third dimension (N) in the shape must be `1`, and the last dimension (D) must be `128` or `64`.
  - Ascend 950DT: Empty tensors are supported. The B dimension in the shape must match that of `query` and `key`, or be `1`. The N dimension in the shape must be `1`, and the last dimension (D) must be less than or equal to `1024`.
- **`sin`** (`Tensor`): Required. Sine value tensor for rotary position embedding. The data type can be `float32`, `float16`, or `bfloat16`. The data layout can be ND. The shape has 3 dimensions when `layout` is `TND`, and 4 dimensions in other `layout` scenarios.
  - Atlas inference products, Atlas A2 training products/Atlas A2 inference products, and Atlas A3 training products/Atlas A3 inference products: Empty tensors are not supported. The B dimension in the shape must match that of `query` and `key`. The third dimension (N) in the shape must be `1`, and the last dimension (D) must be `128` or `64`.
  - Ascend 950DT: Empty tensors are supported. The B dimension in the shape must match that of `query` and `key`, or be `1`. The last dimension (D) in the shape must be less than or equal to 1024.
- **`layout`** (`str`): Optional. Tensor layout format. Valid values: `"BSND"`, `"SBND"`, `"BNSD"`, or `"TND"`. Default value: `"BSND"`.
  - Atlas inference products, Atlas A2 training products/Atlas A2 inference products, and Atlas A3 training products/Atlas A3 inference products: 4D tensors in `BSND` layout and 3D tensors in `TND` layout are supported.
  - Ascend 950DT: 4D tensors in `BSND`, `SBND`, and `BNSD` layouts and 3D tensors in `TND` layout are supported.
- **`rotary_mode`** (`str`): Optional. Rotary encoding mode. Valid values: `"half"`, `"quarter"`, or `"interleave"`. Default value: `"half"`.
  - Atlas inference products, Atlas A2 training products/Atlas A2 inference products, and Atlas A3 training products/Atlas A3 inference products: The `"half"` mode is supported.
  - Ascend 950DT: `"half"`, `"interleave"`, and `"quarter"` modes are supported.

## Return Values

- **`query_out`** (`Tensor`): `query` tensor after the in-place update.
- **`key_out`** (`Tensor`): `key` tensor after the in-place update.

## Constraints

- For <term>Atlas inference products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, and <term>Atlas A3 training products/Atlas A3 inference products</term>:
  - When `layout` is `"BSND"`, the first two dimensions (B and S) of the input shapes for `query`, `key`, `cos`, and `sin` must match. When `layout` is `"TND"`, their first dimension (T) must match.
  - The last dimension (D) of the input shapes for `query` and `key` must match, and the last dimension (D) of the input shapes for `cos` and `sin` must match.
  - The input tensors `query`, `key`, `cos`, and `sin` must have the same data type.
  - When `layout` is `"BSND"`, the shape of the input `query` is represented by `(q_b, q_s, q_n, q_d)`, the shape of the input `key` is represented by `(q_b, q_s, k_n, q_d)`, and the shapes of `cos` and `sin` are represented by `(q_b, q_s, 1, cos_d)`. `b` indicates `batch_size`, `s` indicates `seq_length`, `n` indicates `head_num`, and `d` indicates `head_dim`. When `layout` is `"TND"`, the shape of the input `query` is represented by `(q_t, q_n, q_d)`, the shape of the input `key` is represented by `(q_t, k_n, q_d)`, and the shapes of `cos` and `sin` are represented by `(q_t, 1, cos_d)`. `t` indicates the combined axis of `b` and `s`, `n` indicates `head_num`, and `d` indicates `head_dim`.

- <term>Ascend 950DT</term>:
  - For any `layout`, the dimensions of `query` and `key` must match except for the N dimension. The last dimension (D) of the input shapes for `query` and `key` must match. The last dimension (D) of the input shapes for `cos` and `sin` must match, and must be less than or equal to the last dimension (D) of the input shapes for `query` and `key`.
  - The input tensors `query`, `key`, `cos`, and `sin` must have the same data type.
  - When `rotary_mode` is `"half"` or `"interleave"`, the last dimension of the input shape must be divisible by 2. When `rotary_mode` is `"quarter"`, the last dimension of the input shape must be divisible by 4.
- `bfloat16` is not supported on Atlas inference products.

## Example

```python
import torch
import torch_npu

def test_npu_apply_rotary_pos_emb():

    # Fixed parameters
    batch = 1
    seq_len = 64
    num_heads = 8
    head_dim = 64

    # Create input data.
    query = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16).npu()
    key = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16).npu()
    cos = torch.randn(batch, seq_len, 1, head_dim, dtype=torch.float16).npu()
    sin = torch.randn(batch, seq_len, 1, head_dim, dtype=torch.float16).npu()

    # Call the npu_apply_rotary_pos_emb API.
    q_out, k_out = torch_npu.npu_apply_rotary_pos_emb(
        query, key, cos, sin,
        layout="BSND",
        rotary_mode="half"
    )

    print("API: npu_apply_rotary_pos_emb test passed!")
    print(f"Output query: {q_out}")
    print(f"Output key: {k_out}")

if __name__ == "__main__":
    test_npu_apply_rotary_pos_emb()
```
