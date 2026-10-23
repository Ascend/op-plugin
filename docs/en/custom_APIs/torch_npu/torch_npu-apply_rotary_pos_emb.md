# torch_npu.npu_apply_rotary_pos_emb

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->

## Function

Fuses the `query` and `key` operators into a single path to improve inference performance and performs in-place updates during rotary positional embedding computation.

## Prototype

```python
torch_npu.npu_apply_rotary_pos_emb(query, key,  cos, sin, *, layout='BSND', rotary_mode='half') -> (Tensor, Tensor)
```

## Parameters

- **`query`** (`Tensor`): Required. First tensor for rotary position embedding. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`. The data layout can be ND. The shape has 3 dimensions when `layout` is `TND`, and 4 dimensions in other `layout` scenarios.

  <!-- npu="A3,910b,310p" id5 -->
  - Atlas inference products, Atlas A2 products, and Atlas A3 products: Empty tensors are not supported. The last dimension (`D`) of `shape` must be `128` or `64`.
  <!-- end id5 -->
  <!-- npu="950" id6 -->
  - Ascend 950DT products: Empty tensors are supported. The last dimension (`D`) of `shape` must be less than or equal to `1024`.
  <!-- end id6 -->

- **`key`** (`Tensor`): Required. Second tensor for rotary position embedding. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`. The data layout can be ND. The shape has 3 dimensions when `layout` is `TND`, and 4 dimensions in other `layout` scenarios.

  <!-- npu="A3,910b,310p" id7 -->
  - Atlas inference products, Atlas A2 products, and Atlas A3 products: Empty tensors are not supported. The last dimension (`D`) of `shape` must be `128` or `64`.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - Ascend 950DT products: Empty tensors are supported. The last dimension (`D`) of `shape` must be less than or equal to `1024`.
  <!-- end id8 -->

- **`cos`** (`Tensor`): Required. Cosine value tensor for rotary position embedding. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`. The data layout can be ND. The shape has 3 dimensions when `layout` is `TND`, and 4 dimensions in other `layout` scenarios.

  <!-- npu="A3,910b,310p" id9 -->
  - Atlas inference products, Atlas A2 products, and Atlas A3 products: Empty tensors are not supported. The B dimension in the shape must match that of `query` and `key`. The third dimension (N) in the shape must be `1`, and the last dimension (D) must be `128` or `64`.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - Ascend 950DT products: Empty tensors are supported. The B dimension in the shape must match that of `query` and `key`, or be `1`. The N dimension in the shape must be `1`, and the last dimension (D) must be less than or equal to `1024`.
  <!-- end id10 -->

- **`sin`** (`Tensor`): Required. Sine value tensor for rotary position embedding. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`. The data layout can be ND. The shape has 3 dimensions when `layout` is `TND`, and 4 dimensions in other `layout` scenarios.

  <!-- npu="A3,910b,310p" id11 -->
  - Atlas inference products, Atlas A2 products, and Atlas A3 products: Empty tensors are not supported. The B dimension in the shape must match that of `query` and `key`. The third dimension (N) in the shape must be `1`, and the last dimension (D) must be `128` or `64`.
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - Ascend 950DT products: Empty tensors are supported. The B dimension in the shape must match that of `query` and `key`, or be `1`. The last dimension (D) in the shape must be less than or equal to 1024.
  <!-- end id12 -->

- **`layout`** (`str`): Optional. Tensor layout format. Valid values: `"BSND"`, `"SBND"`, `"BNSD"`, or `"TND"`. Default value: `"BSND"`.

  <!-- npu="A3,910b,310p" id13 -->
  - Atlas inference products, Atlas A2 products, and Atlas A3 products: 4D tensors in `BSND` layout and 3D tensors in `TND` layout are supported.
  <!-- end id13 -->
  <!-- npu="950" id14 -->
  - Ascend 950DT products: 4D tensors in `BSND`, `SBND`, and `BNSD` layouts and 3D tensors in `TND` layout are supported.
  <!-- end id14 -->

- **`rotary_mode`** (`str`): Optional. Rotary encoding mode. Valid values: `"half"`, `"quarter"`, or `"interleave"`. Default value: `"half"`.

  <!-- npu="A3,910b,310p" id15 -->
  - Atlas inference products, Atlas A2 products, and Atlas A3 products: The `"half"` mode is supported.
  <!-- end id15 -->
  <!-- npu="950" id16 -->
  - Ascend 950DT products: `"half"`, `"interleave"`, and `"quarter"` modes are supported.
  <!-- end id16 -->

## Return Values

- **`query_out`** (`Tensor`): `query` tensor after the in-place update.
- **`key_out`** (`Tensor`): `key` tensor after the in-place update.

## Constraints

<!-- npu="A3,910b,310p" id17 -->
- <term>Atlas inference products</term>, <term>Atlas A2 products</term>, and <term>Atlas A3 products</term>:
  - When `layout` is `"BSND"`, the first two dimensions (B and S) of the input shapes for `query`, `key`, `cos`, and `sin` must match. When `layout` is `"TND"`, their first dimension (T) must match.
  - The last dimension (D) of the input shapes for `query` and `key` must match, and the last dimension (D) of the input shapes for `cos` and `sin` must match.
  - The input tensors `query`, `key`, `cos`, and `sin` must have the same data type.
  - When `layout` is `"BSND"`, the shape of the input `query` is represented by `(q_b, q_s, q_n, q_d)`, the shape of the input `key` is represented by `(q_b, q_s, k_n, q_d)`, and the shapes of `cos` and `sin` are represented by `(q_b, q_s, 1, cos_d)`. `b` indicates `batch_size`, `s` indicates `seq_length`, `n` indicates `head_num`, and `d` indicates `head_dim`. When `layout` is `"TND"`, the shape of the input `query` is represented by `(q_t, q_n, q_d)`, the shape of the input `key` is represented by `(q_t, k_n, q_d)`, and the shapes of `cos` and `sin` are represented by `(q_t, 1, cos_d)`. `t` indicates the combined axis of `b` and `s`, `n` indicates `head_num`, and `d` indicates `head_dim`.
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>:
  - For any `layout`, the dimensions of `query` and `key` must match except for the N dimension. The last dimension (D) of the input shapes for `query` and `key` must match. The last dimension (D) of the input shapes for `cos` and `sin` must match, and must be less than or equal to the last dimension (D) of the input shapes for `query` and `key`.
  - The input tensors `query`, `key`, `cos`, and `sin` must have the same data type.
  - When `rotary_mode` is `"half"` or `"interleave"`, the last dimension of the input shape must be divisible by 2. When `rotary_mode` is `"quarter"`, the last dimension of the input shape must be divisible by 4.
<!-- end id18 -->
<!-- npu="310p" id19 -->
- Atlas inference products does not support `torch.bfloat16`.
<!-- end id19 -->

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
