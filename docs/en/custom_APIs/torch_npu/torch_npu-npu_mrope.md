# torch_npu.npu_mrope

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>    |    √     |

## Function

- Description: Applies precomputed `sin` and `cos` positional encoding caches to `query` and `key` in inference scenarios to implement rotary positional embeddings (RoPE). Both standard RoPE (using one-dimensional `positions`) and multimodal MRoPE (using two-dimensional `positions` that combine multiple segments of positional encodings according to `mrope_section`) are supported. That is, `cache_mode="default"` enables segmented concatenation, while `cache_mode="interleave"` enables interleaved concatenation.

- Formulas:

  **MRoPE mode**:
  
  The shape of `positions` is `(m, num_tokens)`, where `m` represents the number of elements in `mrope_section`. Valid values are `3` or `4`.
    
  $$
  cosSin[i] = cosSinCache[positions[i]]
  $$

  $$
  cos, sin = cosSin.chunk(2, dim=-1)
  $$

    - When `cache_mode` is `"default"`:
        - When `mrope_section` contains 3 elements:

        $$
        cos0 = cos[0, :, :mropeSection[0]]
        $$

        $$
        cos1 = cos[1, :, mropeSection[0]:(mropeSection[0] + mropeSection[1])]
        $$

        $$
        cos2 = cos[2, :, (mropeSection[0] + mropeSection[1]):(mropeSection[0] + mropeSection[1] + mropeSection[2])]
        $$

        $$
        cos = torch.cat((cos0, cos1, cos2), dim=-1)
        $$

        $$
        sin0 = sin[0, :, :mropeSection[0]]
        $$

        $$
        sin1 = sin[1, :, mropeSection[0]:(mropeSection[0] + mropeSection[1])]
        $$

        $$
        sin2 = sin[2, :, (mropeSection[0] + mropeSection[1]):(mropeSection[0] + mropeSection[1] + mropeSection[2])]
        $$

        $$
        sin= torch.cat((sin0, sin1, sin2), dim=-1)
        $$

        $$
        queryRot = query[..., :rotaryDim]
        $$

        $$
        queryPass = query[..., rotaryDim:]
        $$

        - When `mrope_section` contains 4 elements:

        $$
        cos = torch.cat([m[i]\ for\ i, m\ in\ enumerate(cos.split(mropeSection, dim=-1))], dim=-1)
        $$

        $$
        sin = torch.cat([m[i]\ for\ i, m\ in\ enumerate(sin.split(mropeSection, dim=-1))], dim=-1)
        $$
        
        $$
        queryRot = query[..., :rotaryDim]
        $$

        $$
        queryPass = query[..., rotaryDim:]
        $$

    - When `cache_mode` is `"interleave"`:

        $$
        cosTmp = cos
        $$

        $$
        cos[..., 1:mropeSection[1] * 3:3] = cosTmp[1, ..., 1:mropeSection[1] * 3:3]
        $$

        $$
        cos[..., 2:mropeSection[1] * 3:3] = cosTmp[2, ..., 2:mropeSection[1] * 3:3]
        $$

        $$
        sinTmp = sin
        $$

        $$
        sin[..., 1:mropeSection[1] * 3:3] = sinTmp [1, ..., 1:mropeSection[1] * 3:3]
        $$

        $$
        sin[..., 2:mropeSection[1] * 3:3] = sinTmp [2, ..., 2:mropeSection[1] * 3:3]
        $$

        $$
        queryRot = query[..., :rotaryDim]
        $$

        $$
        queryPass = query[..., rotaryDim:]
        $$

    - When `rotary_mode` is `half` (GPT-NeoX style):

      $$
      x1, x2 = torch.chunk(queryRot, 2, dim=-1)
      $$

      $$
      o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
      $$

      $$
      o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
      $$

      $$
      queryRot = torch.cat((o1, o2), dim=-1)
      $$

      $$
      query = torch.cat((queryRot, queryPass), dim=-1)
      $$

    - When `rotary_mode` is `"interleaved"` (GPT-J style):

      $$
      x1 = queryRot[..., ::2]
      $$

      $$
      x2 = queryRot[..., 1::2]
      $$

      $$
      o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
      $$

      $$
      o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
      $$

      $$
      queryRot = torch.stack((o1, o2), dim=-1)
      $$

      $$
      query = torch.cat((queryRot, queryPass), dim=-1)
      $$

  **RoPE mode**:
  
  The shape of `positions` is [num_tokens]:

  $$
  cosSin[i] = cosSinCache[positions[i]]
  $$

  $$
  cos, sin = cosSin.chunk(2, dim=-1)
  $$

  $$
  queryRot = query[..., :rotaryDim]
  $$

  $$
  queryPass = query[..., rotaryDim:]
  $$

  - When `rotary_mode` is `half` (GPT-NeoX style):

    $$
    x1, x2 = torch.chunk(queryRot, 2, dim=-1)
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.cat((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

  - When `rotary_mode` is `"interleaved"` (GPT-J style):

    $$
    x1 = queryRot[..., ::2]
    $$

    $$
    x2 = queryRot[..., 1::2]
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.stack((o1, o2), dim=-1)
    $$

    $$
    queryOut = torch.cat((queryRot, queryPass), dim=-1)
    $$

  The computation process for `key` must be identical to that of `query`.

## Prototype

```python
torch_npu.npu_mrope(positions, query, key, cos_sin_cache, head_size, *, mrope_section=None, rotary_mode='half', cache_mode='default') -> (Tensor, Tensor)
```

## Parameters

- **`positions`** (`Tensor`): Required. Position indices used to select positional encodings from the cache, $positions$ in the formulas. 
  - RoPE mode: This parameter must be 1D with shape `(num_tokens,)`. The data type can be `int32` or `int64`. The data layout is ND. Non-contiguous tensors are supported. 
  - MRoPE mode: This parameter must be 2D with shape `(3, num_tokens)` or `(4, num_tokens)`. The first dimension must be equal to the number of sections specified by `mrope_section`. 
  - When positional encodings are selected from `cos_sin_cache` using `positions`, each value serves as a cache row index and must be less than the length of dimension 0 `max_seq_len`.
- **`query`** (`Tensor`): Required. Query tensor to which RoPE is applied, $query$ in the formulas. This parameter must be 2D with shape `(num_tokens, num_q_heads * head_size)`. The data type can be `float16`, `bfloat16`, or `float32`. The data layout is ND. Non-contiguous tensors are supported.
- **`key`** (`Tensor`): Required. Key tensor to which RoPE is applied. This parameter must be 2D with shape `(num_tokens, num_k_heads * head_size)`. The data type must be identical to that of `query`. The data layout is ND. Non-contiguous tensors are supported.
- **`cos_sin_cache`** (`Tensor`): Required. Precomputed positional encoding cache, $cosSinCache$ in the formulas. This parameter must be 2D with shape `(max_seq_len, rotary_dim)`. The last dimension is split into cosine and sine halves using `chunk(2, dim=-1)`. Floating-point data types of `query`, `key`, and `cos_sin_cache` must be identical.
- **`head_size`** (`int`): Required. Size of each attention head, indicating the feature dimension of a single attention head.
- **`mrope_section`** (`List[int]`): Optional. Length configuration of each MRoPE segment in the half-rotation dimension, $mropeSection$ in the formulas. If omitted or set to `[0, 0, 0]`, RoPE mode is used. If set to `[16, 24, 24]`, `[24, 20, 20]`, `[8, 12, 12]`, or `[16, 16, 16, 16]`, MRoPE mode is used. In MRoPE scenarios, the number of elements in `mrope_section` must be identical to the number of rows in `positions`.
- **`rotary_mode`** (`str`): Optional. Rotation style. Valid values are `"half"` (NeoX-style) or `"interleaved"` (GPT-J-style). The default value is `half`.
- **`cache_mode`** (`str`): Optional. Concatenation style. Valid values are `"default"` (segmented cosine/sine concatenation) or `"interleave"` (interleaved concatenation). The default value is `default`. In RoPE scenarios, only `"default"` is supported. In MRoPE scenarios, `"default"` or `"interleave"` can be used.

## Return Values

- **`query_out`** (`Tensor`): Query tensor after applying RoPE or MRoPE to `query`, $queryOut$ in the formulas. This parameter must be 2D with shape `(num_tokens, num_q_heads * head_size)`. The data type can be `float16`, `bfloat16`, or `float32`. The data layout is ND. Non-contiguous tensors are supported. The shape and data type must be identical to those of the input `query`.
- **`key_out`** (`Tensor`): Key tensor after applying RoPE or MRoPE to `key`. This parameter must be 2D with shape `(num_tokens, num_k_heads * head_size)`. The data type must be identical to that of the input `key`. The data layout is ND. Non-contiguous tensors are supported. The shape and data type must be identical to those of the input `key`.

## Constraints

- **Dimension**: `rotary_dim` is the size of the last dimension of `cos_sin_cache` and must be less than or equal to `head_size`.
- **Alignment and multiples**:
  - `head_size` must be divisible by `32` for `float16` or `bfloat16`, and must be divisible by `16` for `float32`.
  - `rotary_dim` must be divisible by `32` for `float16` or `bfloat16`, and must be divisible by `16` for `float32`.
- **MRoPE**: The sum of all elements in `mrope_section` must be identical to **`rotary_dim / 2`** (consistent with the cosine/sine segmentation in the half-rotation dimension).
- **cache_mode**: When `mrope_section` is set to `[16, 16, 16, 16]`, only `cache_mode="default"` is supported. That is, interleaved concatenation is not supported.

## Examples

- MRoPE example (three-segment `mrope_section`):

  `positions` is a 2D tensor with shape `(3, num_tokens)`, corresponding to three-segment configurations such as `[16, 24, 24]`. The sum of the three values in `mrope_section` must be identical to `rotary_dim / 2`.

```python
import torch
import torch_npu

num_tokens = 8
num_q_heads = 32
num_kv_heads = num_q_heads
head_size = 128
max_seq_len = num_tokens
rotary_dim = head_size

positions_mrope = torch.arange(num_tokens, dtype=torch.int64).repeat(3, 1).npu()
query = torch.rand(num_tokens, num_q_heads * head_size, dtype=torch.float32).npu()
key = torch.rand(num_tokens, num_kv_heads * head_size, dtype=torch.float32).npu()
cos_sin_cache = torch.rand(max_seq_len, rotary_dim, dtype=torch.float32).npu()

mrope_section_3 = [16, 24, 24]
query_out, key_out = torch_npu.npu_mrope(
  positions_mrope,
  query,
  key,
  cos_sin_cache,
  head_size,
  mrope_section=mrope_section_3,
  rotary_mode="half",
  cache_mode="default",
)
```

- MRoPE example (four-segment `mrope_section`):

  When `positions` is set to `(4, num_tokens)`, it corresponds to a four-segment configuration such as `[16, 16, 16, 16]`. In this configuration, **only `cache_mode="default"` is supported**.

```python
import torch
import torch_npu

num_tokens = 8
num_q_heads = 32
num_kv_heads = num_q_heads
head_size = 128
max_seq_len = num_tokens
rotary_dim = head_size

positions_mrope4 = torch.arange(num_tokens, dtype=torch.int64).repeat(4, 1).npu()
query = torch.rand(num_tokens, num_q_heads * head_size, dtype=torch.float32).npu()
key = torch.rand(num_tokens, num_kv_heads * head_size, dtype=torch.float32).npu()
cos_sin_cache = torch.rand(max_seq_len, rotary_dim, dtype=torch.float32).npu()

mrope_section_4 = [16, 16, 16, 16]
query_out, key_out = torch_npu.npu_mrope(
  positions_mrope4,
  query,
  key,
  cos_sin_cache,
  head_size,
  mrope_section=mrope_section_4,
  rotary_mode="half",
  cache_mode="default",
)
```

- RoPE example:

  When a 1D tensor is used for `positions`, the shape must be `(num_tokens,)`. Setting `mrope_section` to `[0, 0, 0]` disables MRoPE, which must be identical to the default behavior of `torch_npu`.

```python
import torch
import torch_npu

num_tokens = 8
num_q_heads = 32
num_kv_heads = num_q_heads
head_size = 128
max_seq_len = num_tokens
rotary_dim = head_size

positions_1d = torch.arange(num_tokens, dtype=torch.int64).npu()
query = torch.rand(num_tokens, num_q_heads * head_size, dtype=torch.float32).npu()
key = torch.rand(num_tokens, num_kv_heads * head_size, dtype=torch.float32).npu()
cos_sin_cache = torch.rand(max_seq_len, rotary_dim, dtype=torch.float32).npu()

query_out, key_out = torch_npu.npu_mrope(
  positions_1d,
  query,
  key,
  cos_sin_cache,
  head_size,
  mrope_section=[0, 0, 0],
  rotary_mode="half",
  cache_mode="default",
)
```
