# torch_npu.npu_fused_infer_attention_score

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

## Function<a name="en-us_topic_0000001832267082_section14441124184110"></a>

- Adapts to the `FlashAttention` operator in the incremental and full inference scenarios, supporting both full computation (`PromptFlashAttention`) and incremental computation (`IncreFlashAttention`). When `S` of the `query` matrix is `1`, the execution enters the `IncreFlashAttention` branch. Otherwise, it enters the `PromptFlashAttention` branch. The following two scenarios are supported:

- Formulas:

  $$
  attention\_out = softmax \left(scale \cdot (query \cdot key^\top) + atten\_mask \right) \cdot value
  $$

## Prototype<a name="en-us_topic_0000001832267082_section45077510411"></a>

```python
torch_npu.npu_fused_infer_attention_score(
    query, key, value, *,
    pse_shift=None, atten_mask=None,
    actual_seq_lengths=None, actual_seq_lengths_kv=None,
    dequant_scale1=None, quant_scale1=None,
    dequant_scale2=None, quant_scale2=None, quant_offset2=None,
    antiquant_scale=None, antiquant_offset=None,
    block_table=None, query_padding_size=None, kv_padding_size=None,
    key_antiquant_scale=None, key_antiquant_offset=None,
    value_antiquant_scale=None, value_antiquant_offset=None,
    key_shared_prefix=None, value_shared_prefix=None,
    actual_shared_prefix_len=None,
    query_rope=None, key_rope=None, key_rope_antiquant_scale=None,
    num_heads=1, scale=1.0,
    pre_tokens=2147483647, next_tokens=2147483647,
    input_layout="BSH", num_key_value_heads=0,
    sparse_mode=0, inner_precise=0,
    block_size=0, antiquant_mode=0,
    softmax_lse_flag=False,
    key_antiquant_mode=0, value_antiquant_mode=0
) -> (Tensor, Tensor)
```

  > [!NOTE]  
  > The `query, key, value` before `*` are positional parameters (which must be passed in order); those after `*` are keyword arguments (assigned by key-value pairs, optional; if not passed, the default value is used).

## Parameters<a name="en-us_topic_0000001832267082_section112637109429"></a>

### Parameter quick reference by scenario

| Scenario | Required parameters | Key optional parameters | Corresponding constraint section |
| ---- | -------- | ------------ | ------------ |
| Basic Prefill (MHA) | `query, key, value` | `atten_mask, num_heads, scale, input_layout` | [General basic constraints](#base_constraints), [Q_S > 1 constraints](#qs_gt_1) |
| Basic Decode (MHA) | `query, key, value` | `atten_mask, num_heads, scale, input_layout, actual_seq_lengths_kv` | [General basic constraints](#base_constraints), [Q_S = 1 constraints](#qs_eq_1) |
| Decode + PageAttention | `query, key, value, block_table, block_size` | `actual_seq_lengths_kv` | [PageAttention constraints](#pa_constraints) |
| GQA | `query, key, value, num_heads, num_key_value_heads` | - | [General basic constraints](#base_constraints) |
| MLA | `query, key, value, query_rope, key_rope` | `num_heads, scale, input_layout` | [MLA constraints](#mla_constraints) |
| Prefix (shared prefix) | `query, key, value, key_shared_prefix, value_shared_prefix` | `actual_shared_prefix_len` | [Prefix constraints](#prefix_constraints) |
| `torch.int8` full quantization | `query, key, value, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2` | `quant_offset2` | [`torch.int8` quantization constraints](#int8_constraints) |
| `torch.int8` post-quantization | `query, key, value, quant_scale2` | `quant_offset2` | [`torch.int8` quantization constraints](#int8_constraints) |
| Fake quantization (KV dequantized from `torch.int8` to `torch.float16`) | `query, key, value, antiquant_scale` | `antiquant_offset, antiquant_mode` | [Fake-quantization constraints](#pseudo_quant_constraints) |
| Fake-quantization separation mode | `query, key, value, key_antiquant_scale, value_antiquant_scale` | `key_antiquant_mode, value_antiquant_mode` | [Q_S > 1 constraints](#qs_gt_1) / [Q_S = 1 constraints](#qs_eq_1) |
| Left padding | `query, key, value, actual_seq_lengths, query_padding_size` | `kv_padding_size` | [Padding constraints](#padding_constraints) |
| Ring Attention | `query, key, value, softmax_lse_flag=True` | - | - |
| TND layout | `query, key, value, actual_seq_lengths, actual_seq_lengths_kv, input_layout="TND"` | - | [TND layout constraints](#tnd_constraints) |

### Quick reference of value ranges for each axis

> [!NOTE]  
> B = Batch Size, S = Sequence Length, H = Hidden Size, N = Head Num, D = Head Dim (satisfying D = H / N), T = the sum of the sequence lengths of all batches.
>
> Q_S/S1 = the S of `query`, KV_S/S2 = the S of `key/value`, Q_N = num_query_heads, KV_N = `num_key_value_heads`.

| Axis | Meaning | Q_S > 1 upper limit | Q_S = 1 upper limit | Alignment requirement |
| -- | ---- | ------------ | ------------ | -------- |
| B | Batch Size | <= 65536 | <= 65536 | - |
| N (Q_N) | Number of `query` heads | <= 256 | <= 256 | - |
| N (KV_N) | Number of `key/value` heads | <= 256 | <= 256 | - |
| D | Head Dim | <= 512 | <= 512 | `torch.float16`/`torch.bfloat16`: 16-byte alignment, `torch.int8`: 32-byte alignment, `torch_npu.int4`: 64-byte alignment |
| S (Q_S) | `query` sequence length | <= 20971520 (20M) | = 1 | - |
| S (KV_S) | `key/value` sequence length | <= 20971520 (20M) | <= 262144 | - |

### Detailed parameter description

#### Required parameters (positional parameters)

- **`query`** (`Tensor`): Attention Query input. Data format: $ND$. Non-contiguous tensors are not supported.

  <!-- npu="A3,910b" id4 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Data type: `torch.float16` or `torch.bfloat16`.
  <!-- end id4 -->
  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT products</term>: Data type: `torch.float16` or `torch.bfloat16`.
  <!-- end id5 -->

- **`key`** (`Tensor`): Attention Key input. Data type: `torch.float16`, `torch.bfloat16`, `torch.int8`, or `torch_npu.int4` (in a `torch.int32` container, that is, eight `torch_npu.int4` values packed into one `torch.int32`). Data format: $ND$. Non-contiguous tensors are not supported.

- **`value`** (`Tensor`): Attention Value input. Data type: `torch.float16`, `torch.bfloat16`, `torch.int8`, or `torch_npu.int4` (in a `torch.int32` container). Data format: $ND$. Non-contiguous tensors are not supported.

#### Position encoding parameters

- **`pse_shift`** (`Tensor`, optional): Position encoding parameter. Data type: `torch.float16` or `torch.bfloat16` (must satisfy the derivation rules with the type of `query`). Data format: $ND$. Non-contiguous tensors are not supported. Default value: `None`.

  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Data types: When `Q_S` > 1, the shape is (B, Q_N, Q_S, KV_S) or (1, Q_N, Q_S, KV_S); if KV_S is not 32-byte aligned, padding to 32 bytes is recommended. When `Q_S` = 1, the shape is (B, Q_N, 1, KV_S) or (1, Q_N, 1, KV_S); only the D axis divisible by 16 is supported.
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:
    - When Q_S is not 1, `query` must be of the `torch.float16` type when `pse_shift` is of the `torch.float16` type, and `query` must be of the `torch.bfloat16` type when `pse_shift` is of the `torch.bfloat16` type. The input shape must be (B, N, Q_S, KV_S) or (1, N, Q_S, KV_S), where Q_S is the S in the shape of query, and KV_S is the S in the shapes of key and value. When KV_S of `pse_shift` is not aligned to 32, padding to 32 bytes is recommended to improve performance, and the padding values in the extra part are not restricted.
    - When Q_S is 1, `query` must be of the `torch.float16` type when `pse_shift` is of the `torch.float16` type, and `query` must be of the `torch.bfloat16` type when `pse_shift` is of the `torch.bfloat16` type. The input shape must be (B, N, 1, KV_S) or (1, N, 1, KV_S), where N is num_heads, and KV_S is the S in the shapes of key and value. When KV_S of `pse_shift` is not aligned to 32, padding to 32 bytes is recommended to improve performance, and the padding values in the extra part are not restricted.
  <!-- end id7 -->

#### Mask and sequence length parameters

- **`atten_mask`** (`Tensor`, optional): Applies a mask to the Q x K result to indicate whether the correlation between tokens is computed. Data type: `torch.bool`, `torch.int8`, `torch.uint8`. Data layout: $ND$. Non-contiguous tensors are not supported. Default value: None.

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: When `sparse_mode` is 0/1: the shapes (1, Q_S, KV_S), (B, 1, Q_S, KV_S), and (1, 1, Q_S, KV_S) are supported. When `input_layout` is BSH/BSND/BNSD/BNSD_BSND and RoPE is not passed, (B, KV_S) is supported when Q_S=1, and (Q_S, KV_S) is supported when Q_S>1. When `sparse_mode` is 2/3/4: the shape must be (2048, 2048), (1, 2048, 2048), or (1, 1, 2048, 2048), and the user must ensure that it is lower triangular.
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>:
    - When Q_S is not 1, the recommended input shapes are (B, Q_S, KV_S), (1, Q_S, KV_S), (B, 1, Q_S, KV_S), and (1, 1, Q_S, KV_S).
    - When Q_S is 1, the recommended input shapes are (B, 1, KV_S) and (B, 1, 1, KV_S).
    - Here, Q_S is the S in the shape of `query`, and KV_S is the S in the shapes of `key` and `value`. However, if Q_S and KV_S are not aligned to 16 or 32, they can be rounded up to the aligned S.
  <!-- end id9 -->

- **`actual_seq_lengths`** (`List[int]`, optional): Valid sequence length of `query` in each batch. Data type: `torch.int64`. Default value: None (indicating the same as the S of `query`).

  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The valid sequence length of each batch must be less than or equal to the sequence length of the corresponding batch in `query`. If the value has length 1, all batches share it. If the length is greater than or equal to the batch size, the first `batch` values are used. **This parameter must be passed in the TND layout**: each element is the **cumulative sum** of the sequence lengths of the current batch and all preceding batches (each value is greater than or equal to the preceding value and non-negative), and the number of elements equals the batch value (≤ 4096).
  <!-- end id10 -->

- **`actual_seq_lengths_kv`** (`List[int]`, optional): Valid sequence length of `key/value` in each batch. Data type: `torch.int64`. Default value: None (indicating the same as the S of `key/value`).

  <!-- npu="A3,910b" id11 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The constraints are the same as those of `actual_seq_lengths`. This parameter must be passed in the PageAttention scenario.
  <!-- end id11 -->

#### `torch.int8` quantization parameters

- **`dequant_scale1`** (`Tensor`, optional): Dequantization factor after BMM1. pertensor is supported. Data type: `torch.uint64`, `torch.float32`. Default value: None.

- **`quant_scale1`** (`Tensor`, optional): Quantization factor before BMM2. pertensor is supported. Data type: `torch.float32`. Default value: None.

- **`dequant_scale2`** (`Tensor`, optional): Dequantization factor after BMM2. pertensor is supported. Data type: `torch.uint64`, `torch.float32`. Default value: None.

- **`quant_scale2`** (`Tensor`, optional): Quantization factor of the output. pertensor and perchannel are supported.
  - Data type: `torch.float32`, `torch.bfloat16` (both are supported for `torch.bfloat16` input; otherwise only `torch.float32`). Default value: None.
  - In perchannel mode: for BSH output, the product of all dimensions = H, and the recommended shape is (1, 1, H) or (H,). For BNSD output, the product = Q_N x D, and the recommended shape is (1, Q_N, 1, D) or (Q_N, D). For BSND output, the product = Q_N x D, and the recommended shape is (1, 1, Q_N, D) or (Q_N, D).

- **`quant_offset2`** (`Tensor`, optional): Quantization offset of the output. pertensor and perchannel are supported. If passed, the type and shape must be consistent with those of `quant_scale2`. Data type: `torch.float32`, `torch.bfloat16`. Default value: None (equivalent to 0).

#### Fake-quantization parameters

- **`antiquant_scale`** (`Tensor`, optional): Fake-quantization factor, supporting pertensor, perchannel, or pertoken. Data type: `torch.float16`, `torch.bfloat16` (only `torch.float16` when Q_S >= 2; when Q_S = 1, perchannel uses the same type as `query`, and pertoken uses `torch.float32`). Default value: None. The KV fake-quantization separation mode is recommended (`key_antiquant_scale` and `value_antiquant_scale`).

  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>: This parameter is not supported.
  <!-- end id12 -->

- **`antiquant_offset`** (`Tensor`, optional): Fake-quantization offset, supporting pertensor, perchannel, or pertoken. The constraints are the same as those of `antiquant_scale` (the shape must be consistent). Default value: None. It is not passed (None) in symmetric quantization and must be passed in asymmetric quantization.

  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT products</term>: This parameter is not supported.
  <!-- end id13 -->

- **`antiquant_mode`** (`int`, optional): Fake-quantization mode. 0=perchannel (including pertensor), 1=pertoken. Default value: 0.

  <!-- npu="A3,910b" id14 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: This parameter has no effect when Q_S ≥ 2. When Q_S = 1, passing a value other than 0 or 1 causes an execution error.
  <!-- end id14 -->
  <!-- npu="950" id15 -->
  - <term>Ascend 950PR&950DT products</term>: The values 0 and 1 are supported.
  <!-- end id15 -->

#### PageAttention parameters

- **`block_table`** (`Tensor`, optional): Block mapping table of the KV storage in PageAttention. Data type: `torch.int32`. Data layout: $ND$. It must be 2D: [B, maxBlockNumPerSeq]. The validity of the blockid is guaranteed by the user. Passing this parameter enables PageAttention. Default value: None.

- **`block_size`** (`int`, optional): Maximum number of tokens per block in PageAttention, passed as a Python `int`. Q_S > 1: minimum 128, maximum 512, and must be a multiple of 128. Q_S = 1: non-zero value, maximum 512. `torch.float16`/`torch.bfloat16` requires 16-byte alignment, and `torch.int8` requires 32-byte alignment. The value 128 is recommended. Default value: 0 (PageAttention is disabled).

  <!-- npu="950" id16 -->
  - <term>Ascend 950PR&950DT products</term>: Q_S = 1: when the input types of key and value are `torch_npu.int4 (torch.int32)`, alignment to 64 is required.
  <!-- end id16 -->

#### Padding parameters

- **`query_padding_size`** (`Tensor`, optional): Number of right-aligned paddings per batch of `query`. Data type: `torch.int64`. It takes effect only when Q_S > 1. It must be enabled together with `actual_seq_lengths`; otherwise, right padding is used by default. Default value: None.

- **`kv_padding_size`** (`Tensor`, optional): Number of right-aligned paddings per batch of `key` or `value`. Data type: `torch.int64`. If the value is less than 0, it is set to 0. It must be enabled together with `actual_seq_lengths_kv`; otherwise, right padding is used by default. Default value: None.

#### KV fake-quantization separation parameters

- **`key_antiquant_scale`** (`Tensor`, optional): Dequantization factor of `key` in the KV fake-quantization separation mode. Data type: `torch.float16`, `torch.bfloat16`, `torch.float32`. The perchannel, pertensor, pertoken, pertensor+perhead, pertoken+perhead, and pertoken+PA modes are supported. It must be empty or non-empty at the same time as `value_antiquant_scale`. Default value: None.

- **`key_antiquant_offset`** (`Tensor`, optional): Dequantization offset of `key` in the KV fake-quantization separation mode. The constraints are the same as those of `key_antiquant_scale`. This parameter and `value_antiquant_offset` must either both be empty or both be non-empty. Default value: `None`.

- **`value_antiquant_scale`** (`Tensor`, optional): Dequantization factor of `value` in the KV fake-quantization separation mode. The constraints are the same as those of `key_antiquant_scale`. Default value: None.

- **`value_antiquant_offset`** (`Tensor`, optional): Dequantization offset of `value` in the KV fake-quantization separation mode. The constraints are the same as those of `key_antiquant_scale`. Default value: None.

- **`key_antiquant_mode`** (`int`, optional): Fake-quantization mode of `key`. The value range is 0-5. For details, see the following table. Default value: 0.

- **`value_antiquant_mode`** (`int`, optional): Fake-quantization mode of `value`. The value range is 0-5. For details, see the following table. Default value: 0.

  | Value | Mode | Description |
  | -- | ---- | ---- |
  | 0 | perchannel | perchannel mode (including pertensor). Default value. |
  | 1 | pertoken | pertoken mode. |
  | 2 | pertensor + perhead | pertensor combined with perhead mode. |
  | 3 | pertoken + perhead | pertoken combined with perhead mode. |
  | 4 | pertoken + page attention | pertoken combined with the page attention mode for managing scale/offset. |
  | 5 | pertoken + perhead + page attention | pertoken combined with perhead and using the page attention mode for managing scale/offset. |

> [!NOTE]  
>
><!-- npu="A3,910b" id47 -->
> - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: When Q_S >= 2, only 0 and 1 are supported; when Q_S = 1, 0 to 5 are supported.
><!-- end id47 -->
><!-- npu="950" id48 -->
> - <term>Ascend 950PR&950DT products</term>: `key_antiquant_mode` supports the values 0, 1, 2, 3, 4, and 5. `value_antiquant_mode`: when Q_S >= 2, the values 0, 1, 2, 3, 4, and 5 are supported; when Q_S = 1, the values 0, 1, 2, 3, 4, and 5 are supported. Except in scenarios where `key_antiquant_mode` is 0 and `value_antiquant_mode` is 1, `key_antiquant_mode` and `value_antiquant_mode` must be consistent.
><!-- end id48 -->

#### Prefix (shared prefix) parameters

- **`key_shared_prefix`** (`Tensor`, optional): Shared prefix part of Key. Data type: `torch.float16`, `torch.bfloat16`, `torch.int8`. Data layout: $ND$. Non-contiguous tensors are not supported. The first dimension (batch) of the shape must be 1. It must be empty or non-empty at the same time as `value_shared_prefix`. Default value: None.

- **`value_shared_prefix`** (`Tensor`, optional): Shared prefix part of Value. The constraints are the same as those of `key_shared_prefix`. Default value: None.

- **`actual_shared_prefix_len`** (`List[int]`, optional): Valid sequence length of the shared prefix. Data type: `torch.int64`. The shape is [1], and the value must not be greater than the S of prefix. Default value: None (indicating the same as the S of prefix).

#### MLA (Multi-head Latent Attention) parameters

- **`query_rope`** (`Tensor`, optional): RoPE information of Query in MLA. Data type: `torch.float16`, `torch.bfloat16`. Data layout: $ND$, non-contiguous tensors are not supported. D in the shape is 64, and the other dimensions are consistent with those of `query`. It must be configured at the same time as `key_rope` or not configured at all. Default value: None.

- **`key_rope`** (`Tensor`, optional): RoPE information of Key in MLA. The constraints are the same as those of `query_rope` (D in the shape is 64, and the other dimensions are consistent with those of `key`). Default value: None.

- **`key_rope_antiquant_scale`** (`Tensor`, optional): Reserved parameter, not used currently. Use the default value.

#### Other scalar parameters

- **`num_heads`** (`int`, optional): Number of heads of `query`, passed as a Python `int`. Default value: 1.

  <!-- npu="A3,910b" id17 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: In BNSD scenarios, it must be consistent with the N axis of `query`.
  <!-- end id17 -->
  <!-- npu="950" id18 -->
  - <term>Ascend 950PR&950DT products</term>: In BNSD, BSND, BNSD_BSND, and TND scenarios, it must be consistent with the N axis of `query`.
  <!-- end id18 -->

- **`input_layout`** (`str`, optional): Data layout format of the input `query/key/value`. **When the format contains an underscore**, the left side of the underscore is the input layout and the right side is the output layout, and the operator automatically converts the layout. For details, see the following table. Default value: "BSH".

  | `input_layout` | Input shape (`query`) | Input shape (`key/value`) | Output shape | Applicable scenario |
  | ------------ | ------------------- | ----------------------- | ---------- | -------- |
  | "BSH" | (B, Q_S, H) | (B, KV_S, H) | (B, Q_S, H) | General, H = N x D |
  | `"BSND"` | (B, Q_S, Q_N, D) | (B, KV_S, KV_N, D) | (B, Q_S, Q_N, D) | General, N/D separated |
  | `"BNSD"` | (B, Q_N, Q_S, D) | (B, KV_N, KV_S, D) | (B, Q_N, Q_S, D) | General, N axis first |
  | `"BNSD_BSND"` | (B, Q_N, Q_S, D) | (B, KV_N, KV_S, D) | (B, Q_S, Q_N, D) | Only when Q_S > 1 |
  | `"BSH_NBSD"` | (B, Q_S, H) | (B, KV_S, H) | (B, Q_N, Q_S, D) | Layout conversion |
  | `"BSND_NBSD"` | (B, Q_S, Q_N, D) | (B, KV_S, KV_N, D) | (B, Q_N, Q_S, D) | Layout conversion |
  | `"BNSD_NBSD"` | (B, Q_N, Q_S, D) | (B, KV_N, KV_S, D) | (B, Q_N, Q_S, D) | Only when Q_S > 1 and Q_S <= 16 |
  | `"TND"` | (T, Q_N, D) | (T, KV_N, D) | (T, Q_N, D) | Variable-length sequences, T = sum of seqlens |
  | `"TND_NTD"` | (T, Q_N, D) | (T, KV_N, D) | (Q_N, T, D) | Variable-length sequences + transpose |
  | `"NTD_TND"` | (Q_N, T, D) | (KV_N, T, D) | (T, Q_N, D) | Variable-length sequences + transpose |

<!-- npu="950" id46 -->
> [!NOTE]  
>
> - <term>Ascend 950PR&950DT products</term>: BSH, BSND, BNSD, BNSD_BSND, and TND are supported (left padding, tensorlist, and pse are not supported).
<!-- end id46 -->

- **`scale`** (`float`, optional): Scaling factor, usually **1/sqrt(D)**. The default value is 1.0. **In most scenarios, it must be set manually; otherwise, the computation result will be incorrect.** Default value: 1.0.

- **`pre_tokens`** (`int`, optional): In sparse computation, the range of association between attention and the preceding tokens. It is invalid when Q_S=1. Default value: 2147483647.

- **`next_tokens`** (`int`, optional): In sparse computation, the range of association between attention and the following tokens. It is invalid when Q_S=1. Default value: 2147483647.

- **`num_key_value_heads`** (`int`, optional): Number of heads in `key/value`, passed as a Python `int`, used for GQA. 0 indicates that it is equal to the number of heads of `query`. `num_heads` must be divisible by `num_key_value_heads` (that is, `num_heads` / `num_key_value_heads` must be an integer), and the ratio must be <= 64. Default value: 0.

  <!-- npu="A3,910b" id19 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: In BSND/BNSD/BNSD_BSND scenarios, it must be consistent with the N axis of `key/value`.
  <!-- end id19 -->
  <!-- npu="950" id20 -->
  - <term>Ascend 950PR&950DT products</term>: In BSND/BNSD/BNSD_BSND/TND scenarios, it must be consistent with the N axis of `key/value`.
  <!-- end id20 -->

- **`sparse_mode`** (`int`, optional): Sparse mode. For details, see the following table. Default value: 0.

  | `sparse_mode` | Mode | Description | `atten_mask` requirement |
  | ----------- | ---- | ---- | --------------- |
  | 0 | defaultMask | Default mode. When `atten_mask` is not passed, no mask is applied; when it is passed, the complete mask matrix is used, and the part between `pre_tokens` and `next_tokens` participates in the computation. This parameter is invalid when Q_S=1 and no rope is used. | shape: (Q_S, KV_S) or (1, Q_S, KV_S), etc. |
  | 1 | allMask | The complete `atten_mask` matrix (S1 x S2) must be passed. `pre_tokens` and `next_tokens` are ignored. | shape: (Q_S, KV_S) or (1, Q_S, KV_S), etc. |
  | 2 | leftUpCausal | Upper-left causal mask. The optimized `atten_mask` matrix must be passed. | shape: (2048, 2048) or (1, 2048, 2048) or (1, 1, 2048, 2048) |
  | 3 | rightDownCausal | Lower-right causal mask (lower triangle divided by the right vertex). The optimized mask must be passed. | shape: (2048, 2048) or (1, 2048, 2048) or (1, 1, 2048, 2048) |
  | 4 | band | Band mode mask. The optimized `atten_mask` matrix must be passed. | shape: (2048, 2048) or (1, 2048, 2048) or (1, 1, 2048, 2048) |
  | 5 | prefix | **Not supported currently** | - |
  | 6 | global | **Not supported currently** | - |
  | 7 | dilated | **Not supported currently** | - |
  | 8 | block_local | **Not supported currently** | - |

> [!NOTE]  
> When Q_S = 1 and no rope input is used, the `sparse_mode` parameter is invalid.

- **`inner_precise`** (`int`, optional): Precision/performance mode. For details, see the following table. Default value: 0.

  | `inner_precise` | Precision mode | Invalid-row correction | Applicable scenario |
  | ------------- | -------- | ---------- | -------- |
  | 0 | High precision | No | Default, precision first |
  | 1 | High performance | No | Performance first |
  | 2 | High precision | Yes | Improves precision when the mask contains a row that is all 1s |
  | 3 | High performance | Yes | Improves precision when the mask contains a row that is all 1s (performance slightly decreases) |

> [!NOTE]
>
> `torch.bfloat16` and `torch.int8` do not distinguish high precision from high performance (bit0 is invalid). Invalid-row correction takes effect for `torch.float16`, `torch.bfloat16`, and `torch.int8`. When the mask contains a row that is all 1s, precision may be lost; you can try configuring 2 or 3. When Q_S = 1, only 0 and 1 are supported. If the operator can determine that invalid rows exist (for example, `sparse_mode`=3 and Q_S > KV_S), invalid-row computation is enabled automatically.

- **`softmax_lse_flag`** (`bool`, optional): Whether to additionally output softmax_lse (used for Ring Attention distributed merging). Default value: False.

## Return Values

- **`attention_out`** (`Tensor`): Attention output.
  - Data type: `torch.float16`, `torch.bfloat16`, `torch.int8`. Data layout: $ND$. The D dimension is consistent with the D of `value`, and the other dimensions are consistent with the shape of `query`.

- **`softmax_lse`** (`Tensor`): Log-sum-exp value of softmax (used to merge the results of each device in Ring Attention distributed scenarios).
  - When `softmax_lse_flag=True`: when `input_layout` is TND/NTD_TND, the shape is (T, Q_N, 1) and the dtype is `torch.float32`. In other cases, the shape is (B, Q_N, Q_S, 1) and the dtype is `torch.float32`.
  - When `softmax_lse_flag=False`: the shape is (1,), and the value is a Tensor of 0s.

  Ring Attention multi-device merge formula:

  $$
  O_{global} = \frac{e^{lse_1 - lse_{max}} \cdot O_1 + e^{lse_2 - lse_{max}} \cdot O_2}{e^{lse_1 - lse_{max}} + e^{lse_2 - lse_{max}}}
  $$

  Where $lse_{max} = max(lse_1, lse_2)$.

## Constraints

### <a id="base_constraints"></a>General basic constraints

- This API supports inference scenarios and graph mode.
- When used with PyTorch, the CANN and PyTorch versions must match.
- **Handling of empty inputs**: If `query` is empty, an empty result is returned directly. If `query` is not empty and `key/value` is empty (S2=0), `attention_out` is returned as all 0s according to the corresponding shape. If `attention_out` is empty, an empty result is returned.
- The shapes of `key` and `value` must be completely consistent. In non-contiguous scenarios, the batch in the `key/value` tensorlist can only be 1, the count must equal the B of `query`, and N and D must be equal.
- The default value of `scale` is 1.0. It should usually be set to **1/sqrt(D)**; otherwise, the computation result is incorrect.

### <a id="int8_constraints"></a>`torch.int8` quantization constraints

Constraints on the combination of parameters and input/output data types:

| Input dtype | Output dtype | Must be passed | Can be passed | Must not be passed |
| ---------- | ---------- | -------- | -------- | -------- |
| `torch.int8` | `torch.int8` | `dequant_scale1, quant_scale1, dequant_scale2, quant_scale2` | `quant_offset2` | — |
| `torch.int8` | `torch.float16` | `dequant_scale1, quant_scale1, dequant_scale2` | — | `quant_offset2, quant_scale2` |
| `torch.float16`/`torch.bfloat16` | `torch.int8` | `quant_scale2` | `quant_offset2` | `dequant_scale1, quant_scale1, dequant_scale2` |

- `quant_scale2` and `quant_offset2` support pertensor or perchannel, and the dtype supports `torch.float32`/`torch.bfloat16`. When `quant_offset2` is passed, its type and shape must be consistent with those of `quant_scale2`.
- `input_layout` supports only BSH, BNSD, BSND, and BNSD_BSND.

<!-- npu="A3,910b" id21 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - When the output is `torch.int8` and the parameters `quant_scale2` and `quant_offset2` are in perchannel mode: left padding, Ring Attention, and non-32-byte-aligned D are not supported.
  - When the output is `torch.int8`, sparse=band with negative `pre_tokens`/`next_tokens` is not supported currently.
  - When the output is `torch.int8` and `quant_offset2` is not None, if the matrix contains rows that do not participate in computation, an interception is triggered (external post-quantization is recommended).
<!-- end id21 -->

### <a id="pseudo_quant_constraints"></a>Fake-quantization constraints (antiquant_scale / antiquant_offset)

- Only the kv dtype of `torch.int8` is supported.
- perchannel: the shape is (2, KV_N, 1, D), (2, KV_N, D), or (2, H), and the dtype is the same as that of `query`. `antiquant_mode`=0.
- pertensor: the shape is (2,), and the dtype is the same as that of `query`. `antiquant_mode`=0.
- pertoken: the shape is (2, B, KV_S), and the dtype is fixed to `torch.float32`. `antiquant_mode`=1.
- Symmetric quantization: `antiquant_offset`=None; asymmetric quantization: both must exist at the same time.
- The operator automatically determines the mode based on the shape dim (pertensor when dim=1; otherwise perchannel).
- **The KV fake-quantization separation mode is recommended** (use `key_antiquant_scale` and `value_antiquant_scale` instead of `antiquant_scale` and `antiquant_offset`).

### <a id="pa_constraints"></a>PageAttention constraints

**Enabling condition**: `block_table` exists and is valid. Once enabled, the `input_layout` parameter of `key/value` becomes invalid, and `key/value` are arranged in contiguous memory indexed by `block_table`.

| Constraint | Q_S > 1 | Q_S = 1 |
| ------ | ------- | ------- |
| `block_size` | Minimum 128, maximum 512, and must be a multiple of 128 | <li><term>Atlas A2 products</term>, <term>Atlas A3 products</term>: non-zero, maximum 512, and 128 is recommended.</li><li><term>Ascend 950PR&950DT products</term>: when the input types of `key` and `value` are `torch.float16` or `torch.bfloat16`, alignment to 16 is required; when the input types of `key` and `value` are `torch.int8`, alignment to 32 is required, and 128 is recommended; when the input types of `key` and `value` are `torch_npu.int4 (torch.int32)`, alignment to 64 is required.</li> |
| kv dtype | `torch.float16`, `torch.bfloat16` | <li><term>Atlas A2 products</term>, <term>Atlas A3 products</term>: `torch.float16`, `torch.bfloat16`, `torch.int8`.</li><li><term>Ascend 950PR&950DT products</term>: `torch.float16`, `torch.bfloat16`, `torch.int8`, `torch_npu.int4` (`torch.int32`)</li> |
| `query` `torch.int8` | Not supported | Not supported |
| kv cache layout | For BSH/BSND, only BnBsH; for BNSD/TND, both BnBsH and BnNBsD are supported | Same as the left |
| `actual_seq_lengths_kv` must be passed | Yes | Yes |
| `block_table` is 2D | [B, >=maxBlockNumPerSeq] | [B, >=maxBlockNumPerSeq] |
| Lower limit of blocknum | >= the sum of the numbers of blocks of each batch | Same as the left |
| Supports tensorlist | No | No |
| Supports left padding | No | No |
| Supports fake quantization | No | Yes (`torch.int8` kv) |
| Performance tip | BnNBsD is better than BnBsH | BnNBsD is better than BnBsH |
| When H exceeds 65535 | Enable GQA or use BnNBsD | Enable GQA or use BnNBsD |

- When PageAttention is enabled, the following scenarios require KV_S >= maxBlockNumPerSeq x `block_size`:
  - `atten_mask` is passed (for example, shape (B, 1, Q_S, KV_S))
  - `pse_shift` is passed (for example, shape (B, Q_N, Q_S, KV_S))

  <!-- npu="A3,910b" id22 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - When Q_S = 1 and `atten_mask` is enabled (`sparse_mode` is not 2/3/4): the last dimension of `atten_mask` >= the second dimension of `block_table` x `block_size`.
    - When Q_S = 1 and `pse_shift` is enabled: the last dimension of `pse_shift` >= the second dimension of `block_table` x `block_size`.
    - Q_S = 1 fake-quantization pertoken mode: the S of the shape of `antiquant_scale`/`antiquant_offset` >= this value.
  <!-- end id22 -->
  <!-- npu="950" id23 -->
  - <term>Ascend 950PR&950DT products</term>:
    - When per-token combined with per-head mode is enabled: the shapes of the two parameters are both (B, N, S), the data type is fixed to `torch.float32`, and this is supported when the data types of key and value are `torch.int8` or `torch_npu.int4` (`torch.int32`).
  <!-- end id23 -->

### <a id="mla_constraints"></a>MLA constraints

The MLA scenario is entered when `query_rope` and `key_rope` are input.

- `query_rope` and `key_rope` must be configured at the same time or not configured at all.
- The dtype/format of `query_rope` is consistent with that of `query`; the dtype/format of `key_rope` is consistent with that of `key`.
- The D of `query` supports only 512 or 128.

**When D = 512:**

<!-- npu="A3,910b" id24 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - sparse supports 0/3/4.
  - `query_rope`: the N of `query` is required to be 1/2/4/8/16/32/64/128, D in the shape is 64, and the other dimensions are consistent with those of `query`.
  - `key_rope`: key is required to have N=1 and D=512, D in the shape is 64, and the other dimensions are consistent with those of `key`.
  - `key/value/key_rope` support $ND$ or $NZ$. In the $NZ$ format: the shape of `torch.float16`/`torch.bfloat16` is [blockNum, KV_N, D/16, blockSize, 16]; the shape of `torch.int8` is [blockNum, KV_N, D/32, blockSize, 32].
  - `input_layout`: BSH, BSND, BNSD, BNSD_NBSD, BSND_NBSD, BSH_NBSD, TND, TND_NTD.
  - PageAttention is supported (`block_size`: a multiple of 16 and <= 1024).
  - Not supported: softmax_lse, left padding, tensorlist, pse, prefix, fake quantization, full quantization, and post-quantization.
<!-- end id24 -->
<!-- npu="950" id25 -->
- <term>Ascend 950PR&950DT products</term>:
  - sparse: when Q_S equals 1, only sparse=0 without mask passed is supported; when Q_S is greater than 1, sparse=0 without mask passed or sparse=3 with mask passed is supported;
  - The `input_layout` of `key, value` supports the $ND$ format.
  - `input_layout`: BNSD, BSND, BSH, TND.
  - In this scenario, when `input_layout` is BNSD/BSH/BSND, only the `actual_seq_lengths_kv` parameter is supported. When `input_layout` is TND, the `actual_seq_lengths` and `actual_seq_lengths_kv` parameters must be passed.
<!-- end id25 -->

**When D = 128:**

<!-- npu="A3,910b" id26 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - `input_layout`: BSH, BSND, TND, BNSD, NTD, BSH_BNSD, BSND_BNSD, BNSD_BSND, NTD_TND.
  - D in the shapes of `query_rope`/`key_rope` is 64, and the other dimensions are unchanged.
  - Left padding, tensorlist, pse, prefix, fake quantization, full quantization, and post-quantization are not supported.
  - The other constraints are the same as those in TND/NTD_TND scenarios.
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950PR&950DT products</term>:
  - When `query_rope` is configured, b, n, and s in the shape of `query_rope` must be consistent with those of `query`, and d is 64.
  - When `key_rope` is configured, b, n, and s in the shape of `key_rope` must be consistent with those of `key`, and d is 64.
  - `input_layout`: BSH, BSND, BNSD, BNSD_BSND, TND.
  - Page attention, prefix, fake quantization, full quantization, and post-quantization are not supported.
  - When kv is a tensorlist, b in the shape of `key_rope` must be consistent with the tensorlist length; n and s must be equal to the n and s of each tensor in the tensorlist; and d is 64.
<!-- end id27 -->

### <a id="prefix_constraints"></a>Prefix (shared prefix) constraints

- `key_shared_prefix` and `value_shared_prefix` must both be empty or both be non-empty.
- When non-empty: both have the same dimensions and dtype as `key`/`value`.
- The first dimension (batch) of the shape of `key_shared_prefix` is 1. In BNSD/BSND, the N and D axes are consistent with those of `key`; in BSH, H is consistent with that of `key`. The same applies to `value_shared_prefix`. The S values of the two must be equal.
- When `actual_shared_prefix_len` exists: shape=[1], and the value <= the S of prefix.
- The S of prefix + the S of `key/value` <= the original S limit of `key/value`.
- Not supported: PageAttention, left padding, tensorlist, and all-qkv `torch.int8`.
- When sparse=0/1 and `atten_mask` is passed: S2 >= `actual_shared_prefix_len` + the S of `key`.

### <a id="padding_constraints"></a>Padding constraints

**query left padding (only Q_S > 1)**:

- Transfer start = Q_S - `query_padding_size` - `actual_seq_lengths` (must be >= 0).
- Transfer end = Q_S - `query_padding_size` (must be <= Q_S).
- When `kv_padding_size` < 0, it is set to 0.
- It must be enabled together with `actual_seq_lengths`.
- PageAttention is not supported.

**kv left padding**:

- Transfer start = KV_S - `kv_padding_size` - `actual_seq_lengths_kv` (must be >= 0 when Q_S > 1; when Q_S = 1, all 0s are returned if it is < 0).
- Transfer end = KV_S - `kv_padding_size` (must be <= KV_S when Q_S > 1; when Q_S = 1, all 0s are returned if it is < 0).
- It must be enabled together with `actual_seq_lengths_kv`.
- PageAttention is not supported.

<!-- npu="A3,910b" id28 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - The scenario where Q is `torch.bfloat16`/`torch.float16` and KV is `torch_npu.int4` is not supported.
<!-- end id28 -->

### <a id="tnd_constraints"></a>TND layout constraints

- `actual_seq_lengths` and `actual_seq_lengths_kv` must be passed. The number of elements is the batch value (<= 4096). Each element value is a cumulative sum (the latter value >= the former value).

<!-- npu="A3,910b" id29 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  
  Comprehensive restrictions when `input_layout` is TND, TND_NTD, or NTD_TND:

  - **When query D = 512**:

    - sparse: 0/3/4.
    - TND and TND_NTD are supported.
    - PageAttention is supported (the length of `actual_seq_lengths_kv` = the number of kv batches, and the value <= KV_S).
    - `query` N = 1/2/4/8/16/32/64/128, kv N = 1.
    - `query_rope` and `key_rope` are non-empty (D=64).
    - Not supported: left padding, tensorlist, pse, prefix, fake quantization, and full quantization.
  - **When query D is not 512**:
    - Without rope: Q_D, K_D, V_D = 128, or Q_D, K_D = 192 and V_D = 128 or 192 (NTD does not support V_D=192; NTD_TND requires Q_D, K_D = 128 or 192 and V_D = 128).
    - With rope: Q_D, K_D, V_D = 128.
    - TND, NTD, and NTD_TND are supported.
    - For PageAttention, only a `block_size` that is a multiple of 16 and <= 1024 is supported.
    - Not supported: left padding, tensorlist, pse, prefix, fake quantization, full quantization, and post-quantization.
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950PR&950DT products</term>:
  - Left padding, tensorlist, pseType=0, and prefix are not supported.
<!-- end id30 -->

### <a id="qs_gt_1"></a>Q_S > 1 (Prefill) exclusive constraints

- **B axis**: <= 65536.

  <!-- npu="A3,910b" id31 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: When the D axis is not 32-byte aligned, <= 128.
  <!-- end id31 -->
  <!-- npu="950" id32 -->
  - <term>Ascend 950PR&950DT products</term>: A B axis of less than or equal to 65536 is supported.
  <!-- end id32 -->

- **N axis**: <= 256.
- **D axis**: <= 512. In BSH/BSND, N x D < 65535 is recommended.
- **S axis**: <= 20971520 (20M).
- **D alignment**:

  <!-- npu="A3,910b" id33 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: `torch.int8` must be 32-byte aligned, `torch_npu.int4` must be 64-byte aligned, and `torch.float16`/`torch.bfloat16` must be 16-byte aligned.
  <!-- end id33 -->
  <!-- npu="950" id34 -->
  - <term>Ascend 950PR&950DT products</term>: Non-quantization scenarios: the types of `query`, `key`, and `value` are all `torch.float16` or `torch.bfloat16`, and D axes from 1 to 512 are all supported. Fake-quantization scenarios: the type of `query` is `torch.float16` or `torch.bfloat16`, and the types of `key` and `value` are `torch.int8`/`torch_npu.int4` (`torch.int32`). When the types of `key` and `value` are `torch_npu.int4` (`torch.int32`), the D axis supports only 64 alignment (`torch.int32` supports only D aligned to 8).
  <!-- end id34 -->

<!-- npu="950" id35 -->
- **actual_seq_lengths**:

  - <term>Ascend 950PR&950DT products</term>: This parameter must be a non-negative number, and its meaning and interception conditions differ for different `input_layout` values. Generally, this parameter is optional, and its length is 1 or greater than or equal to the Batch value of `query`. The value in this parameter represents the actual length of each batch and must not be greater than Q_S. When `input_layout` is TND, this parameter must be passed, and the number of elements of this parameter is used as the Batch value. The value of each element indicates the sum of the seqlens of the current batch and all preceding batches. Therefore, the value of the latter element must be greater than or equal to that of the former element, and negative values must not occur.
<!-- end id35 -->

<!-- npu="950" id36 -->
- **actual_seq_lengths_kv**:

  - <term>Ascend 950PR&950DT products</term>: When passed in, this parameter must be a non-negative number, and its meaning and interception conditions differ for different `input_layout` values. Generally, this parameter is optional, and the valid seqlenKv of each batch in this parameter must not be greater than the seqlenKv of the corresponding batch in `key/value`. When the passed length of this parameter is 1, each batch uses the same seqlenKv; when the passed length is greater than or equal to Batch, the first Batch values of seqlenKv are used. Other lengths are not supported. When the `input_layout` of `key/value` is TND/TND_NTD, this parameter must be passed. In non-PA scenarios, the b-th value indicates the accumulated length of the S axes of the first b batches, and the values must be arranged in increasing order (greater than or equal to the previous value). The number of elements of this parameter represents the total number of batches. In PA scenarios, its length equals the Batch value of `key/value`, it represents the actual length of each batch, and the values must not be greater than KV_S.
<!-- end id36 -->
- **Long-sequence timeout risk**: When B x N x S x D is too large, a timeout may occur (aicore timeout/trap error). Splitting S is recommended. Typical high-risk scenarios:
  - B=1, Q_N=20, Q_S=2097152, D=256, KV_N=1, KV_S=2097152
  - B=1, Q_N=2, Q_S=20971520, D=256, KV_N=2, KV_S=20971520
  - B=20, Q_N=1, Q_S=2097152, D=256, KV_N=1, KV_S=2097152
  - B=1, Q_N=10, Q_S=2097152, D=512, KV_N=1, KV_S=2097152
- **sparse_mode**: Only 0/1/2/3/4 are supported. Other values cause an error.
  - 0: `atten_mask`=None, or `pre_tokens` and `next_tokens` are ignored during left padding.
  - 2/3/4: the mask shape must be (S,S) or (1,S,S) or (1,1,S,S), where S=2048, and the user guarantees the lower triangle.
  - 1/2/3: `pre_tokens` and `next_tokens` are ignored.
- **kvCache dequantization composite parameters**: Only `torch.int8` dequantized to `torch.float16` is supported. The data range of `key/value` x the data range of `antiquant_scale` must be within (-1, 1) (high-performance mode); otherwise, high-precision mode is required.
- **pse_shift**: It can be used when the dtype of `query` is `torch.float16`/`torch.bfloat16`/`torch.int8`. In the case of `torch.float16` `query` + `pse_shift`, the high-precision mode is forced. Q_S >= the S of `query`, and KV_S >= the S of `key`. For prefix, KV_S >= `actual_shared_prefix_len` + the S of `key`.
- **GQA fake quantization + KV NZ format**:

  <!-- npu="A3,910b" id37 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - `query`: `torch.bfloat16`, `key/value`: `torch.int8`, D=128, Q_S=1~16.
    - `input_layout`: BSH/BSND/BNSD.
    - Only PageAttention is supported (`block_size`=128 or 512).
    - `key/value` NZ format: [blockNum, KV_N, D/32, blockSize, 32].
    - dtype of `key/value_antiquant_scale`: `torch.bfloat16` in perchannel mode and `torch.float32` in pertoken mode.
    - Only the KV separation mode is supported. offset, rope, left padding, tensorlist, pse, prefix, and post-quantization are not supported.
    - Supported combinations of `num_heads` and `num_key_value_heads`: (10,1), (64,8), (80,8), and (128,16).
    - MTP=0: `sparse_mode`=0 without mask. MTP>0 and <16: `sparse_mode`=3 with the optimized mask (2048x2048).
    - Only the high-performance mode is supported.
  <!-- end id37 -->

<!-- npu="950" id38 -->
- **kv fake-quantization parameter separation**:

  - <term>Ascend 950PR&950DT products</term>:
    - When fake-quantization parameters and KV separation quantization parameters are passed at the same time, the KV separation quantization parameters take precedence.
    - Except in scenarios where `key_antiquant_mode`=0 and `value_antiquant_mode`=1, the values of `key_antiquant_mode` and `value_antiquant_mode` must be consistent.
    - `key_antiquant_scale` and `value_antiquant_scale` must either both be empty or both be non-empty; `key_antiquant_offset` and `value_antiquant_offset` must either both be empty or both be non-empty.
    - When neither `key_antiquant_scale` nor `value_antiquant_scale` is empty, their shapes must be consistent. When neither `key_antiquant_offset` nor `value_antiquant_offset` is empty, their shapes must be consistent.
    - When `key_antiquant_mode`=0 and `value_antiquant_mode`=1, the values of `key_antiquant_mode` and `value_antiquant_mode` do not need to be consistent. In all other scenarios, they must be consistent.
    - `key_antiquant_scale` and `value_antiquant_scale` must either both be empty or both be non-empty; `key_antiquant_offset` and `value_antiquant_offset` must either both be empty or both be non-empty.
    - When neither `key_antiquant_scale` nor `value_antiquant_scale` is empty, except in scenarios where `key_antiquant_mode`=0 and `value_antiquant_mode`=1, their shapes must be consistent.
    - When neither `key_antiquant_offset` nor `value_antiquant_offset` is empty, except in scenarios where `key_antiquant_mode`=0 and `value_antiquant_mode`=1, their shapes must be consistent.
    - The quantization modes for managing scale/offset are as follows:

      > [!NOTE]  
      > Note that scale and offset refer to the `key_antiquant_scale`, `value_antiquant_scale`, `key_antiquant_offset`, and `value_antiquant_offset` parameters.

      | Quantization Mode | scale and offset conditions in this scenario | `key` and `value` conditions in this scenario |
      | --- | --- | --- |
      | per-channel mode | The shapes of the two parameters support (N, 1, D), (N, D), (H), (1, N, 1, D), (1, N, D), and (1, H), and the data type is the same as that of query. | Supported when the data types of key, value are `torch_npu.int4` (`torch.int32`) or `torch.int8`. |
      | per-token mode | The shapes of the two parameters are both (B, S), and the data type is fixed to `torch.float32`. | Supported when the data types of `key, value` are `torch_npu.int4` (`torch.int32`) or `torch.int8`. |
      | per-tensor mode | The shapes of the two parameters are both (1), and the data type is the same as that of query. | Supported when the data types of key, value are `torch_npu.int4` (`torch.int32`) or `torch.int8`. |
      | per-tensor combined with per-head mode | The shapes of the two parameters are both (N), and the data type is the same as that of query. | Supported when the data types of key, value are `torch_npu.int4` (`torch.int32`) or `torch.int8`. |
      | per-token combined with per-head mode | The shapes of the two parameters are both (B, N, S), and the data type is fixed to `torch.float32`. | Supported when the data types of key, value are `torch_npu.int4` (`torch.int32`) or `torch.int8`. |
      | per-token combined with the page attention mode | The shapes of the two parameters are both (blocknum, blocksize), and the data type is fixed to `torch.float32`. | Supported when the data types of `key, value` are `torch.int8`. |
      | per-token combined with per head and using the page attention mode | The shapes of the two parameters are both (blocknum, N, blocksize), and the data type is fixed to `torch.float32`. | Supported when the data types of `key, value` are `torch.int8`. |
      | `key` supports per-channel combined with `value` supports per-token mode | For `key` supporting per-channel, the shapes of the two parameters can be (N, 1, D), (N, D), (H), (1, N, 1, D), (1, N, D), or (1, H), and the data type is the same as that of `query`. For `value` supporting per-token, the shapes of the two parameters are both (B, S) and the data type is fixed to `torch.float32`. | Supported when the data types of `key, value` are `torch_npu.int4` (`torch.int32`) or `torch.int8`. |
<!-- end id38 -->

### <a id="qs_eq_1"></a>Q_S = 1 (Decode) exclusive constraints

- **B axis**: <= 65536.
- **N axis**: <= 256.
- **D axis**: <= 512.
- **S axis** (KV): <= 262144.
- **All-qkv `torch.int8`**: Not supported.
- **`torch_npu.int4` (`torch.int32`) fake quantization**:
  - PyTorch graph calls support only the case where KV `torch_npu.int4` is concatenated into a `torch.int32` input (generated through dynamic_quant is recommended: 1 `torch.int32` contains 8 `torch_npu.int4` values).
  - KV N, D, or H is 1/8 of the actual value (the same applies to prefix).
  - `torch_npu.int4` supports only D aligned to 64 bytes (`torch.int32` supports D aligned to 8 bytes).
- **actual_seq_lengths**:

  <!-- npu="A3,910b" id39 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: This parameter is invalid when the layout is not TND.
  <!-- end id39 -->
  <!-- npu="950" id40 -->
  - <term>Ascend 950PR&950DT products</term>: This parameter does not take effect when no rope parameter is input and the layout is not TND. It takes effect when a rope parameter is input. When passed in, it must be a non-negative number, and its meaning and interception conditions differ for different `input_layout` values. Generally, this parameter is optional, and its length is 1 or greater than or equal to the Batch value of `query`. The value in this parameter represents the actual length of each batch and must not be greater than Q_S. When `input_layout` is TND, this parameter must be passed, and the number of elements of this parameter is used as the Batch value. The value of each element indicates the sum of the seqlens of the current batch and all preceding batches. Therefore, the value of the latter element must be greater than or equal to that of the former element, and negative values must not occur.
  <!-- end id40 -->

- **actual_seq_lengths_kv**:

  <!-- npu="A3,910b" id41 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: When the layout is not TND, the passed length is 1/>=batch; the passed value <= the corresponding KV_S.
  <!-- end id41 -->
  <!-- npu="950" id42 -->
  - <term>Ascend 950PR&950DT products</term>: This parameter must be a non-negative number, and its meaning and interception conditions differ for different input_layout values. Generally, this parameter is optional, and the valid Sequence Length of each batch in this parameter must not be greater than the seqlenKv of the corresponding batch in key/value. When the passed length of this parameter is 1, each batch uses the same seqlenKv; when the passed length is greater than or equal to Batch, the first Batch values of seqlenKv are used. Other lengths are not supported. When input_layout is TND/TND_NTD, this parameter must be passed. In non-PA scenarios, the b-th value indicates the accumulated length of the S axes of the first b batches, and the values must be arranged in increasing order (greater than or equal to the previous value). The number of elements of this parameter represents the total number of batches. In PA scenarios, its length equals the Batch value of key/value, it represents the actual length of each batch, and the values must not be greater than KV_S.
  <!-- end id42 -->

- **pse_shift**: the dtype is consistent with that of `query`.

  <!-- npu="A3,910b" id43 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Only D-axis divisibility by 16 is supported.
  <!-- end id43 -->

- **kv left padding**: When the transfer start/end is < 0, all 0s are returned. It must be enabled together with `actual_seq_lengths_kv`.
- **`torch_npu.int4` kv**:

  <!-- npu="A3,910b" id44 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Left padding and page attention are not supported (Q: `torch.bfloat16`/`torch.float16` + KV: `torch_npu.int4`).
  <!-- end id44 -->

<!-- npu="950" id45 -->
- **kv fake-quantization parameter separation**:

  - <term>Ascend 950PR&950DT products</term>:
    - Except in scenarios where `key_antiquant_mode`=0 and `value_antiquant_mode`=1, the values of `key_antiquant_mode` and `value_antiquant_mode` must be consistent.
    - `key_antiquant_scale` and `value_antiquant_scale` must either both be empty or both be non-empty; `key_antiquant_offset` and `value_antiquant_offset` must either both be empty or both be non-empty.
    - When neither `key_antiquant_scale` nor `value_antiquant_scale` is empty, except in scenarios where `key_antiquant_mode`=0 and `value_antiquant_mode`=1, their shapes must be consistent. When neither `key_antiquant_offset` nor `value_antiquant_offset` is empty, except in scenarios where `key_antiquant_mode`=0 and `value_antiquant_mode`=1, their shapes must be consistent.
    - The `torch_npu.int4` (`torch.int32`) fake-quantization scenario does not support post-quantization.
    - Usage constraints for the `pse_shift` feature:

      > [!NOTE]  
      > Note that scale and offset refer to `key_antiquant_scale`, `value_antiquant_scale`, `key_antiquant_offset`, and `value_antiquant_offset`.
      > When Q_S equals 1, the quantization modes listed in the following table are supported.

      | Quantization Mode | scale and offset conditions in this scenario | `key` and `value` conditions in this scenario |
      | --- | --- | --- |
      | per-channel mode | The shapes of the two parameters support (1, N, 1, D), (1, N, D), and (1, H), and the data type is the same as that of `query`. | Supported when the data types of `key` and `value` are `torch_npu.int4` (`torch.int32`) or `torch.int8`. |
      | per-tensor mode | The shapes of the two parameters are both (1), and the data type is the same as that of `query`. | Supported when the data types of `key` and `value` are `torch.int8`. |
      | per-token mode | The shapes of the two parameters are both (1, B, S), and the data type is fixed to `torch.float32`. | Supported when the data types of `key` and `value` are `torch_npu.int4` (`torch.int32`) or `torch.int8`. |
      | per-tensor combined with per-head mode | The shapes of the two parameters are both (N), and the data type is the same as that of `query`. | Supported when the data types of `key` and `value` are `torch.int8`. |
      | per-token combined with per-head mode | The shapes of the two parameters are both (B, N, S), and the data type is fixed to `torch.float32`. | Supported when the data types of `key` and `value` are `torch_npu.int4` (`torch.int32`) or `torch.int8`. |
      | per-token combined with the page attention mode | The shapes of the two parameters are both (blocknum, blocksize), and the data type is fixed to `torch.float32`. | Supported when the data types of `key` and `value` are `torch.int8`. |
      | per-token combined with per head and using the page attention mode | The shapes of the two parameters are both (blocknum, N, blocksize), and the data type is fixed to `torch.float32`. | Supported when the data types of `key` and `value` are `torch.int8`. |
      | `key` supports per-channel combined with `value` supports per-token mode | For `key` supporting per-channel, the shapes of the two parameters can be (1, N, 1, D), (1, N, D), or (1, H), and the data type is the same as that of `query`. For `value` supporting per-token, the shapes of the two parameters are both (1, B, S) and the data type is fixed to `torch.float32`. | Supported when the data types of `key` and `value` are `torch_npu.int4` (`torch.int32`) or `torch.int8`. |

<!-- end id45 -->
## Examples<a name="en-us_topic_0000001832267082_section14459801435"></a>

### Example 1: MHA + Prefill (basic scenario)

The simplest full computation call: single batch, 8 heads, D=128, BNSD layout.

```python
import torch
import torch_npu
import math

# ===== Data preparation =====
B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N, KV_S = 8, 1024

# Shapes of query/key/value: (B, N, S, D) - BNSD layout
q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()

# scale must be set to 1/sqrt(D)
scale = 1.0 / math.sqrt(D)

# actual_seq_lengths: length 1 means all batches share it
# actual_seq_lengths_kv: valid length of key/value
act_seq_len = [Q_S]
act_seq_len_kv = [KV_S]

# ===== Call the operator =====
# MHA: num_heads == num_key_value_heads (or num_key_value_heads is not passed and defaults to 0)
out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    actual_seq_lengths=act_seq_len,
    actual_seq_lengths_kv=act_seq_len_kv,
    num_heads=Q_N,
    input_layout="BNSD",
    scale=scale,
    pre_tokens=65535,
    next_tokens=65535
)

print(out.shape)  # torch.Size([1, 8, 164, 128])
```

> [!NOTE]  
>
> - `scale=1/√128`: Required. Controls the scaling of QK^T before softmax. An incorrect setting causes an abnormal attention distribution.
>
> - `pre_tokens=65535, next_tokens=65535`: Indicates that the attention range is not limited (equivalent to bidirectional attention).
>
> - `input_layout="BNSD"`: Indicates that the input is (B,N,S,D).
>
> - This scenario falls under Q_S > 1 and uses the PromptFlashAttention branch.

### Example 2: GQA + Decode + PageAttention

Incremental inference scenario of grouped-query attention + KV cache + PageAttention.

```python
import torch
import torch_npu
import math

# ===== Data preparation =====
B = 1
Q_N = 8           # query heads
KV_N = 2          # key/value heads (GQA: Q_N / KV_N = 4)
Q_S = 1           # the query length in the decode phase = 1
D = 128
KV_S = 2048       # total KV cache length
block_size = 128
# Calculate the required number of blocks
max_block_num = (KV_S + block_size - 1) // block_size  # = 16

# BSH layout: query is 3D (B, S, H_q), where H_q = Q_N * D
H_q = Q_N * D    # = 1024
H_kv = KV_N * D  # = 256
q = torch.randn(B, Q_S, H_q, dtype=torch.float16).npu()
# KV cache: BnBsH format (blocknum, block_size, H_kv)
k = torch.randn(max_block_num, block_size, H_kv, dtype=torch.float16).npu()
v = torch.randn(max_block_num, block_size, H_kv, dtype=torch.float16).npu()

# block_table: (B, max_block_num), stores block IDs
block_table = torch.arange(max_block_num, dtype=torch.int32).unsqueeze(0).npu()
scale = 1.0 / math.sqrt(D)

# In PageAttention scenarios, actual_seq_lengths_kv must be passed
act_seq_len_kv = [KV_S]

# ===== Call the operator =====
out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    actual_seq_lengths_kv=act_seq_len_kv,
    num_heads=Q_N,
     num_key_value_heads=KV_N,   # GQA: specify the number of KV heads
    input_layout="BSH",          # use BSH when the kv cache is in BnBsH format
    scale=scale,
    block_table=block_table,
    block_size=block_size
)

print(out.shape)  # torch.Size([1, 1, 1024])
```

> [!NOTE]  
>
> - `num_key_value_heads=2`: Enables GQA. num_heads / num_key_value_heads = 4. When this parameter is not passed, it defaults to 0 (that is, MHA). For details, see [Parameter description](#detailed-parameter-description).
>
> - `block_table` + `block_size`: Passing them enables PageAttention. For the kv cache layout, see [PageAttention constraints](#pa_constraints).
>
> - `input_layout="BSH"`: BSH must be used when the kv cache is in BnBsH format. If the kv cache is in BnNBsD format and query is BNSD, BNSD can be used.
>
> - `actual_seq_lengths_kv`: Must be passed in PageAttention scenarios. For the corresponding constraints, see [PageAttention constraints](#pa_constraints).
>
> - In this scenario, Q_S=1, and the IncreFlashAttention branch is used. For the constraints, see [Q_S = 1 constraints](#qs_eq_1).

### Example 3: MHA + Causal Mask (with atten_mask)

device='npu:0', dtype=torch.float16)

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N, KV_S = 8, 164

q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()

# Lower-triangle causal mask: (1, Q_S, KV_S)
mask = torch.tril(torch.ones(Q_S, KV_S, dtype=torch.bool)).unsqueeze(0).npu()

scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    atten_mask=mask,
    num_heads=Q_N,
    input_layout="BNSD",
    scale=scale,
    sparse_mode=0,     # defaultMask: use the complete mask matrix
    pre_tokens=65535,
    next_tokens=65535
)

print(out.shape)  # torch.Size([1, 8, 164, 128])
```

> [!NOTE]  
>
> - `atten_mask`: shape (1, Q_S, KV_S), of the `torch.bool` type. For more mask shape options, see [atten_mask parameter description](#detailed-parameter-description).
>
> - `sparse_mode=0`: defaultMask mode, in which `pre_tokens`/`next_tokens` control the computation range. Here it is complete bidirectional attention.
>
> - Alternatively, `sparse_mode=3` (rightDownCausal) + the 2048x2048 optimized mask can be used.

### Example 4: MLA + Decode (Multi-head Latent Attention)

MLA inference scenario for DeepSeek-type models.

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 16, 1, 512   # in MLA scenarios, D=512
KV_N, KV_S = 1, 2048
rope_D = 64                       # the RoPE dimension is fixed at 64

# query: (B, Q_N, Q_S, D)
q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
# key/value: (B, KV_N, KV_S, D)
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()

# RoPE tensors: D=64 in the shape, and the other dimensions are consistent with q/k
q_rope = torch.randn(B, Q_N, Q_S, rope_D, dtype=torch.float16).npu()
k_rope = torch.randn(B, KV_N, KV_S, rope_D, dtype=torch.float16).npu()

scale = 1.0 / math.sqrt(D + rope_D)  # the effective dimension in MLA scenarios = D + 64

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    query_rope=q_rope,
    key_rope=k_rope,
    num_heads=Q_N,
    num_key_value_heads=KV_N,
    input_layout="BNSD",
    scale=scale,
    sparse_mode=0
)

print(out.shape)  # torch.Size([1, 16, 1, 512])
```

> [!NOTE]  
>
> - `query_rope` + `key_rope`: Passing them enters the MLA branch, and they must be passed together. The RoPE dimension is fixed at 64. Their dtype/format must be consistent with that of query/key.
>
> - D=512: The D of MLA prefill scenarios is 512 or 128. When D=512, PageAttention can be enabled and the NZ format is supported. For details about the constraints, see [MLA constraints](#mla_constraints).
>
> - This scenario does not support: softmax_lse, left padding, pse, prefix, fake quantization, and full quantization.

### Example 5: `torch.int8` post-quantization

Only the output is `torch.int8` quantized (post-training quantization), and the input remains `torch.float16`.

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N, KV_S = 8, 1024

q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()

# pertensor quantization factor: shape (1, 1, 1) or a scalar shape
quant_scale2 = torch.tensor([0.01], dtype=torch.float32).npu()
# If quant_offset2 is not passed, the default is 0

scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    quant_scale2=quant_scale2,
    num_heads=Q_N,
    input_layout="BNSD",
    scale=scale,
    pre_tokens=65535,
    next_tokens=65535
)

print(out.dtype)  # torch.int8
print(out.shape)  # torch.Size([1, 8, 164, 128])
```

> [!NOTE]  
>
> - `quant_scale2`: Passing it triggers post-quantization (the output is `torch.int8`). If `quant_offset2` is not passed, the offset is 0.
>
> - In `torch.int8` output scenarios, `dequant_scale1`/`quant_scale1`/`dequant_scale2` must **not** be passed at the same time. For details about the constraints, see [`torch.int8` quantization constraints](#int8_constraints).
>
> - For perchannel quantization, the shape (Q_N, D) is recommended. For details, see [quant_scale2 parameter description](#detailed-parameter-description).

### Example 6: `torch.int8` full quantization

The input is `torch.int8`, which is internally dequantized to `torch.float16` for computation, and the output is then quantized to `torch.int8`.

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N, KV_S = 8, 1024

# int8 input (ensure that the D axis is 32-byte aligned)
q = torch.randint(-128, 127, (B, Q_N, Q_S, D), dtype=torch.int8).npu()
k = torch.randint(-128, 127, (B, KV_N, KV_S, D), dtype=torch.int8).npu()
v = torch.randint(-128, 127, (B, KV_N, KV_S, D), dtype=torch.int8).npu()

# Dequantization factor after BMM1 (pertensor)
deq_scale1 = torch.tensor([0.01], dtype=torch.float32).npu()
# Quantization factor before BMM2
quant_scale1 = torch.tensor([0.02], dtype=torch.float32).npu()
# Dequantization factor after BMM2
deq_scale2 = torch.tensor([0.03], dtype=torch.float32).npu()
# Output quantization factor
quant_scale2 = torch.tensor([0.04], dtype=torch.float32).npu()

scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    dequant_scale1=deq_scale1,
    quant_scale1=quant_scale1,
    dequant_scale2=deq_scale2,
    quant_scale2=quant_scale2,
    num_heads=Q_N,
    input_layout="BNSD",
    scale=scale,
    pre_tokens=65535,
    next_tokens=65535
)

print(out.dtype)  # torch.int8
```

> [!NOTE]
>
> - The four quantization parameters must **exist at the same time**. `quant_offset2` is optional; if not passed, the offset is 0.
>
> - The D axis must be 32-byte aligned. input_layout supports only BSH/BNSD/BSND/BNSD_BSND.
>
> - For details about the constraints, see [`torch.int8` quantization constraints](#int8_constraints).

### Example 7: fake-quantization (KV split mode) + Decode

key/value are `torch.int8` and are converted to `torch.float16` by perchannel dequantization factors for computation. The recommended KV split mode is used.

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 8, 1, 128
KV_N, KV_S = 2, 2048

q = torch.randn(B, Q_N, Q_S, D, dtype=torch.bfloat16).npu()
k = torch.randint(-128, 127, (B, KV_N, KV_S, D), dtype=torch.int8).npu()
v = torch.randint(-128, 127, (B, KV_N, KV_S, D), dtype=torch.int8).npu()

# KV split perchannel quantization factors: shape (1, KV_N, 1, D), (1, KV_N, D), or (1, H)
key_antiquant_scale = torch.ones(1, KV_N, 1, D, dtype=torch.bfloat16).npu() * 0.01
value_antiquant_scale = torch.ones(1, KV_N, 1, D, dtype=torch.bfloat16).npu() * 0.01
# Symmetric quantization: do not pass offset

scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    key_antiquant_scale=key_antiquant_scale,
    value_antiquant_scale=value_antiquant_scale,
    key_antiquant_mode=0,     # perchannel
    value_antiquant_mode=0,
    num_heads=Q_N,
    num_key_value_heads=KV_N,
    input_layout="BNSD",
    scale=scale
)

print(out.dtype)   # torch.bfloat16
print(out.shape)   # torch.Size([1, 8, 1, 128])
```

> [!NOTE]  
>
> - The KV split mode is used (preferred over the `antiquant_scale` and `antiquant_offset` mode).
>
> - `key_antiquant_mode=0`: perchannel mode. Here, Q_S=1 + query `torch.bfloat16` + kv `torch.int8` + key/value_antiquant_scale `torch.bfloat16`, which satisfies the kv fake-quantization parameter separation condition in [Q_S = 1 constraints](#qs_eq_1).
>
> - When `key_antiquant_scale` and `value_antiquant_scale` are passed together with `antiquant_scale`, the KV split parameters take precedence.

### Example 8: Prefix + Prefill (shared prefix)

System prompt/shared prefix scenario: the prefix part serves as the common KV and is concatenated with the KV of each request for computation.

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N, KV_S = 8, 1024
prefix_S = 32

q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.float16).npu()

# Shared prefix: batch=1, S=prefix_S, N/D consistent with k/v
key_prefix = torch.randn(1, KV_N, prefix_S, D, dtype=torch.float16).npu()
value_prefix = torch.randn(1, KV_N, prefix_S, D, dtype=torch.float16).npu()

scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score(
    q, k, v,
    key_shared_prefix=key_prefix,
    value_shared_prefix=value_prefix,
    num_heads=Q_N,
    input_layout="BNSD",
    scale=scale,
    pre_tokens=65535,
    next_tokens=65535
)

print(out.shape)  # torch.Size([1, 8, 164, 128])
```

> [!NOTE]  
>
> - `key_shared_prefix` + `value_shared_prefix`: They must be passed together. The first dimension (batch) of the shape is 1. For the constraints, see [Prefix constraints](#prefix_constraints).
>
> - If S2 (KV_S + prefix_S) exceeds the original KV_S limit, an error is reported. If `atten_mask` is passed, S2 must be greater than prefix_S + KV_S.
>
> - PageAttention, left padding, tensorlist, and all-qkv `torch.int8` are not supported.

### Example 9: ACL Graph mode (torch_npu.npu.NPUGraph)

Capture the compute graph with NPUGraph to enable graph-mode execution and parameter updates.

```python
import torch
import torch_npu
import math

# ===== 1. Data preparation =====
B, Q_N, Q_S, D = 1, 8, 164, 128
KV_S = 1024
scale = 1.0 / math.sqrt(D)

q = torch.randn(B, Q_N, Q_S, D, dtype=torch.float16).npu()
k = torch.randn(B, Q_N, KV_S, D, dtype=torch.float16).npu()
v = torch.randn(B, Q_N, KV_S, D, dtype=torch.float16).npu()

# ===== 2. Define the model =====
class Model(torch.nn.Module):
    def __init__(self, num_heads, scale, pre_tokens=65535, next_tokens=65535):
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens

    def forward(self, q, k, v):
        return torch_npu.npu_fused_infer_attention_score(
            q, k, v,
            num_heads=self.num_heads,
            input_layout="BNSD",
            scale=self.scale,
            pre_tokens=self.pre_tokens,
            next_tokens=self.next_tokens
        )[0]  # take only the first return value

# ===== 3. Graph capture =====
model = Model(Q_N, scale).npu()
graph = torch_npu.npu.NPUGraph()

with torch_npu.npu.graph(graph):
    graph_out = model(q, k, v)

print(f"Graph capture complete, output shape: {graph_out.shape}")

# ===== 4. Graph replay (input data update) =====
# Generate new input data (content changes, shape unchanged)
q_new = torch.randn_like(q)
k_new = torch.randn_like(k)
v_new = torch.randn_like(v)

# Important: use copy_ to update the data at the memory address captured at capture time
q.copy_(q_new)
k.copy_(k_new)
v.copy_(v_new)

# Replay the graph
graph.replay()
print(f"Graph replay complete (input data updated), output shape: {graph_out.shape}")

# ===== 5. Parameter update (model configuration change) =====
# Note: Changes to pre_tokens/next_tokens require recapturing the graph
new_pre_tokens = 32768
model_updated = Model(Q_N, scale, new_pre_tokens, 32768).npu()

graph_updated = torch_npu.npu.NPUGraph()
with torch_npu.npu.graph(graph_updated):
    updated_out = model_updated(q, k, v)

graph_updated.replay()
print(f"Graph replay complete (pre_tokens={new_pre_tokens}), output shape: {updated_out.shape}")

# ===== 6. Result verification =====
with torch.no_grad():
    single_out = model(q, k, v)

match = torch.allclose(graph_out, single_out, atol=1e-2)
print(f"\nGraph vs Eager match: {'Yes' if match else 'No'}")
print(f"  graph output shape: {graph_out.shape}")
print(f"  single op output shape: {single_out.shape}")
```

## References

- **BMM1 (Batch Matrix Multiply 1)**: The first matrix multiplication, which computes the attention score: $attention_score = query \cdot key^{\top}$.
- **BMM2 (Batch Matrix Multiply 2)**: The second matrix multiplication, which applies the attention weights to value: $attention_out = atten_prob \cdot value$, where $atten_prob = softmax(\cdot)$.
- **Ring Attention**: A distributed attention algorithm that splits long sequences into blocks across multiple devices and computes attention block by block through ring communication to reduce the memory usage of a single device. On a single device, it degrades to standard attention. Multi-device merging relies on the LSE value output when `softmax_lse_flag=True`. For details, see [Return values](#return-values).
