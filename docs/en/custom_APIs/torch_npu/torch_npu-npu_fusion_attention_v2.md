# torch_npu.npu_fusion_attention_v2

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

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

- API function: In training scenarios, this API uses the FlashAttention algorithm to implement fused self-attention computation. It supports both non-varlen (BSH, SBH, BSND, BNSD) and varlen (TND) scenarios, and supports features such as RoPE extension inputs for query/key (`query_rope`, `key_rope`), per-attention-head sink bias, and GQA (grouped-query attention).
- Formulas:
  - When pse_type is 1 (default), the forward attention formula is as follows:

    $$
    attention\_out = Dropout(Softmax(Mask(scale*(query*key^T + pse), atten\_mask)), keep\_prob)*value
    $$

  - When pse_type has other values, the formula is as follows:

    $$
    attention\_out = Dropout(Softmax(Mask(scale*(query*key^T) + pse), atten\_mask), keep\_prob)*value
    $$

  - When query_rope and key_rope are passed (varlen scenario), the formula is as follows:

    $$
    attention\_out = Dropout(Softmax(Mask(scale*(query*key^T + query\_rope*key\_rope^T) + pse), atten\_mask), keep\_prob)*value
    $$

  - When sink is passed, the sink bias is added to the computation logic. The softmax_max and softmax_sum computations are mainly modified:

    $$
    S = Q * K^{T}
    $$

    $$
    m = max(sink, max(S))
    $$

    $$
    Attention = \frac{e^{S - m} * V}{\sum e^{S-m} + e^{sink - m}}
    $$

<!-- npu="A3,910b" id12 -->
> [!NOTE]
> On <term>Atlas A3 products</term> and <term>Atlas A2 products</term>, query_rope/key_rope take effect only in varlen scenarios (input_layout is TND), and actual_seq_qlen and actual_seq_kvlen must be passed with valid values.
<!-- end id12 -->

## Prototype

```python
torch_npu.npu_fusion_attention_v2(query, key, value, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None, query_rope=None, key_rope=None, scale=1., keep_prob=1., pre_tokens=2147483647, next_tokens=2147483647, inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False, pse_type=1, q_start_idx=None, kv_start_idx=None, softmax_layout="", sink=None, dropout_mask=None, seed=0, offset=0) -> (Tensor, Tensor, Tensor, Tensor, int, int, int)
```

## Parameters

- **`query`** (`Tensor`): Required. $query$ in the formula. The number of dimensions can be 3 or 4. The data layout can be ND. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32` (for product-specific data type constraints when query_rope/key_rope is passed, see the description of the `query_rope` parameter). It must be the same as the data type of key and value.
- **`key`** (`Tensor`): Required. $key$ in the formula. The number of dimensions can be 3 or 4. The data layout can be ND. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32` (for product-specific data type constraints when query_rope/key_rope is passed, see the description of the `query_rope` parameter). It must be the same as the data type of query and value.
- **`value`** (`Tensor`): Required. $value$ in the formula. The number of dimensions can be 3 or 4. The data layout can be ND. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32` (for product-specific data type constraints when query_rope/key_rope is passed, see the description of the `query_rope` parameter). It must be the same as the data type of query and key.
- **`head_num`** (`int`): Required. Number of heads on a single NPU, that is, the length of the N axis of the input query.
- **`input_layout`** (`str`): Required. Data layout of the input `query`, `key`, and `value`. BSH, SBH, BSND, BNSD, and TND are supported (case-insensitive). When `input_layout` is TND, it is a varlen scenario, and `actual_seq_qlen`/`actual_seq_kvlen` must be passed. Unless otherwise specified, S indicates the sequence length of `query` or `key`/`value`, Sq indicates the sequence length of query, Skv indicates the sequence length of `key`/`value`, and SS indicates Sq\*Skv.
- **`*`**: Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.
- **`pse`** (`Tensor`): Optional. $pse$ in the formula, that is, position encoding. It must be used together with `pse_type`. The default value is None. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The data layout can be ND.
  - In non-varlen scenarios, four-dimensional input is supported, including the BNSS, BN1Skv, and 1NSS formats.
  - If Sq is greater than 1024 in a non-varlen scenario, or in a varlen scenario where Sq and Skv of each batch have the same length and `sparse_mode` is 0, 2, or 3 in a lower-triangular mask scenario, alibi position encoding compression can be enabled. In this case, only the last 1024 rows of the original PSE need to be input for memory optimization, that is, alibi\_compress = ori\_pse[:, :, -1024:, :]. When the parameter differs for each batch, input BNHSkv (H=1024); when it is the same for each batch, input 1NHSkv (H=1024). The applicability of compression in varlen scenarios is as follows:

    <!-- npu="950" id4 -->
    - <term>Ascend 950PR&950DT products</term>: Compression is also supported in varlen scenarios.
    <!-- end id4 -->
    <!-- npu="A3,910b" id13 -->
    - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: Compression is supported only in non-varlen scenarios, and Sq must be greater than 1024, Sq and Skv of each batch must have the same length, and the lower-triangular mask condition must be met.
    <!-- end id13 -->
  - In varlen scenarios, the BNHSkv (H=1024), 1NHSkv (H=1024), and pseTotalLen shapes are supported. pseTotalLen is the sum of the number of pse elements of all batch segments. The number of pse elements of the i-th batch segment is N \* Sq\_i \* Skv\_i.
  - When `pse_type` is 2 or 3, the data type must be `torch.float32`, and the corresponding supported shape range is [B, N] or [N].
- **`padding_mask`** (`Tensor`): Optional. Reserved parameter. It is not used currently. Pass None.
- **`atten_mask`** (`Tensor`): Optional. $atten\_mask$ in the formula. The value 1 (or True) indicates that the position does not participate in the computation (is masked), and the value 0 (or False) indicates that the position participates in the computation (is retained). The data type can be `torch.bool` or `torch.uint8`. The data layout can be ND. The input shape can be in the BNSS, B1SS, 11SS, or SS format. In varlen scenarios, only the SS format is supported, where SS is maxSq and maxSkv respectively.
- **`query_rope`** (`Tensor`): Optional. $query\_rope$ in the formula, that is, the RoPE extension input of query. It must be passed together with `key_rope`. The default value is None. It is supported only in varlen scenarios (input_layout is TND), and `atten_mask` must be passed. The data layout can be ND. The shape type [TND] is supported. The Head-Dim must satisfy (qRoPED == kRoPED), D must be a multiple of 8 and less than or equal to the D of query, key, and value. The supported data types are as follows:

  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT products</term>: The same as `query` (`torch.float16`, `torch.bfloat16`, `torch.float32`).
  <!-- end id5 -->
  <!-- npu="A3,910b" id14 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: Only `torch.bfloat16` is supported.
  <!-- end id14 -->
- **`key_rope`** (`Tensor`): Optional. $key\_rope$ in the formula, that is, the RoPE extension input of key. It must be passed together with `query_rope`. The default value is None. It is supported only in varlen scenarios (input_layout is TND), and `atten_mask` must be passed. The data type, data layout, shape, and Head-Dim constraints are the same as those of `query_rope`.
- **`scale`** (`float`): Optional. Scaling coefficient, used as the scalar value of Muls in the computation flow. The default value is 1.
- **`keep_prob`** (`float`): Optional. Proportion of 1 in Dropout. The value range is (0, 1]. The default value is 1, indicating that all values are retained. When query_rope/key_rope is passed, keep_prob must be 1.
- **`pre_tokens`** (`int`): Optional. Used for sparse computation. It indicates the left boundary of the sliding window. The default value is 2147483647.
- **`next_tokens`** (`int`): Optional. Used for sparse computation. It indicates the right boundary of the sliding window. The default value is 2147483647. For the relationship between the values of `next_tokens` and `pre_tokens` and `atten_mask`, see the `sparse_mode` parameter. If the parameter values are inconsistent with the `atten_mask` distribution, accuracy problems may occur.
- **`inner_precise`** (`int`): Optional. Used to improve precision. The default value is 0. Currently, 0 and 1 are reserved configuration values, and 2 enables invalid row computation. Its function is to avoid precision loss caused by an entire row being masked during computation, but this configuration degrades performance. If the operator can determine that an invalid row scenario exists, it automatically enables invalid row computation, for example, when `sparse_mode` is 3 and Sq > Skv.
- **`prefix`** (`List[int]`): Optional. N value of each batch in the prefix sparse computation scenario. The default value is None. The data type can be `torch.int64`. When Sq > Skv, the N value range of prefix is [0, Skv]. When Sq <= Skv, the N value range of prefix is [Skv-Sq, Skv]. Varlen scenarios do not support non-compressed prefix (that is, sparse_mode=5 is not supported).
- **`actual_seq_qlen`** (`List[int]`): Optional. This parameter must be passed in varlen scenarios. It indicates the cumulative sum length of each S of `query`. The default value is None. The data type can be `torch.int64`. The length range is 1 to 2K, and each element must be less than or equal to 1M. For example, if the actual S length list is [2, 2, 2, 2, 2], pass `actual_seq_qlen` as [2, 4, 6, 8, 10].
- **`actual_seq_kvlen`** (`List[int]`): Optional. This parameter must be passed in varlen scenarios. It indicates the cumulative sum length of each S of `key`/`value`. The default value is None. The data type can be `torch.int64`. The length range is 1 to 2K, and each element must be less than or equal to 1M. For example, if the actual S length list is [2, 2, 2, 2, 2], pass `actual_seq_kvlen` as [2, 4, 6, 8, 10].
- **`sparse_mode`** (`int`): Optional. Sparse mode. The default value is 0. When `atten_mask` is None, the `sparse_mode`, `pre_tokens`, and `next_tokens` parameters do not take effect, and full computation is used. For the principles of different modes, see [References](torch_npu-npu_fusion_attention.md#references). The value scenarios are as follows.

  | sparse_mode | Meaning | Note |
  | --- | --- | --- |
  | 0 | defaultMask mode | - |
  | 1 | allMask mode | - |
  | 2 | leftUpCausal mode | - |
  | 3 | rightDownCausal mode | - |
  | 4 | band mode | - |
  | 5 | prefix non-compressed mode | Not supported in varlen scenarios |
  | 6 | prefix compressed mode | Not supported in varlen scenarios when query_rope/key_rope is passed |
  | 7 | varlen outer-cut scenario, rightDownCausal mode | Supported only in varlen scenarios |
  | 8 | varlen outer-cut scenario, leftUpCausal mode | Supported only in varlen scenarios |

  - Varlen scenarios (input_layout is TND) support the values 0, 1, 2, 3, 4, 6, 7, and 8. Non-varlen scenarios support the values 0 to 6.
  - When the `atten_mask` of the entire network is the same and its shape is smaller than 2048\*2048, the defaultMask mode is recommended to reduce memory usage.

- **`gen_mask_parallel`** (`bool`): Optional. Control switch for the DSA generation of the dropout random number vector mask. The default value is True: parallel computation with the same AI Core. If set to False: serial computation with the same AI Core.
- **`sync`** (`bool`): Optional. When `gen_mask_parallel=True`, controls whether to synchronously wait for dropout mask generation to complete. The default value is False: the dropout mask is generated asynchronously. If set to True: the dropout mask is generated synchronously.
- **`pse_type`** (`int`): Optional. Controls the order of mul and add computations. The default value is 1. The meanings of different values are as follows.

  | pse_type | Meaning | Note |
  | --- | --- | --- |
  | 0 | Externally passed pse is multiplied first and then added | - |
  | 1 | Externally passed pse is added first and then multiplied | Default value |
  | 2 | Internally generated pse is multiplied first and then added | pse must be `torch.float32` with shape [B, N] or [N] |
  | 3 | Internally generated pse is multiplied first, then added, and then sqrt is applied | pse must be `torch.float32` with shape [B, N] or [N] |

  - When `pse_type` is 2 or 3, currently only the case where Sq and Skv have the same length is supported.
  - When query_rope/key_rope is passed, `pse_type` supports only the value 1.

- **`q_start_idx`** (`List[int]`): Optional. Start index in the global sequence of the query in the current block in the outer-cut scenario. The default value is None. The data type can be `torch.int64`.
- **`kv_start_idx`** (`List[int]`): Optional. Start index in the global sequence of the key and value in the current block in the outer-cut scenario. The default value is None. The data type can be `torch.int64`.
- **`softmax_layout`** (`str`): Optional. Controls the data layout of the softmax output (softmax_max and softmax_sum) in TND scenarios. The default value is "". Only "" and "TND" can be passed, and "TND" can be passed only when `input_layout` is TND. The output formats of softmax_max and softmax_sum in TND scenarios are as follows:

  <!-- npu="950" id6 -->
  - <term>Ascend 950PR&950DT products</term>: When "TND" is passed, the softmax output layout is TN8; otherwise, it is BNS8.
  <!-- end id6 -->
  <!-- npu="A3,910b" id18 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: When "TND" is passed, the softmax output layout is TND; otherwise, it is NTD.
  <!-- end id18 -->
- **`sink`** (`Tensor`): Optional. Bias of each attention head. The default value is None. The shape is [head\_num]. The data type can be `torch.float32` only.
- **`dropout_mask`** (`Tensor`): Optional. Externally passed dropout mask. The default value is None. When it is not passed, it is automatically generated by the API according to `keep_prob`, `seed`, and `offset`.
- **`seed`** (`int`): Optional. Seed of the Philox algorithm for DSA generation of the dropout mask. The default value is 0. The seed in the return value is the seed actually used, which can be used for backward computation.
- **`offset`** (`int`): Optional. Offset of the Philox algorithm for DSA generation of the dropout mask. The default value is 0. The offset in the return value is the offset actually used, which can be used for backward computation.

## Return Values

There are 7 outputs in total, with types **Tensor, Tensor, Tensor, Tensor, int, int, int** in order.

- **`attention_out`** (`Tensor`): Final output of the formula. The data type and shape type are the same as those of query (in BNSD scenarios, the output shape is [B, N, Sq, Dv]; in TND scenarios, the output shape is [T, N, Dv], where Dv is the Head-Dim of value).
- **`softmax_max`** (`Tensor`): Max intermediate result of the softmax computation, used for backward computation. The data type is `torch.float32`. In non-varlen scenarios, the shape is [B, N, Sq, 8]. In varlen (TND) scenarios, the shape is [T, N, 8].
- **`softmax_sum`** (`Tensor`): Sum intermediate result of the softmax computation, used for backward computation. The data type is `torch.float32`. In non-varlen scenarios, the shape is [B, N, Sq, 8]. In varlen (TND) scenarios, the shape is [T, N, 8].
- **`softmax_out`** (`Tensor`): Reserved parameter. It is not used currently. An empty Tensor is returned.
- **`seed`** (`int`): Seed of the Philox algorithm for DSA generation of the dropout mask. It returns the seed actually used, which can be used for backward computation.
- **`offset`** (`int`): Offset of the Philox algorithm for DSA generation of the dropout mask. It returns the offset actually used, which can be used for backward computation.
- **`numels`** (`int`): Number of elements in the dropout mask.

## Constraints

- This API is used only in training scenarios.
- This API supports only single-operator mode.
- Graph mode and aclgraph are not supported currently.
- Deterministic computation: The deterministic implementation is used by default. When `keep_prob` is less than 1, the dropout mask is generated from random numbers, and the determinism of the computation result is controlled by the `seed` and `offset` parameters.
- The dimensions of the input `query`, `key`, and `value` must be 3 or 4, and `input_layout` must be the same.
- The data types of the input `query`, `key`, and `value` must be the same. When query_rope/key_rope is passed:

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: The data type is unchanged, and `torch.float16`, `torch.bfloat16`, and `torch.float32` are still supported.
  <!-- end id7 -->
  <!-- npu="A3,910b" id15 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: Only `torch.bfloat16` is supported.
  <!-- end id15 -->
- When `pse` is input, its data type must be the same as that of `query`, `key`, and `value`.
- The shapes of the input `key` and `value` must be the same (except for Head-Dim). There is one exception: in the case of unequal D lengths, the D of `key` and the D of `value` can be different, but the D of `value` must be smaller than the D of `key`. The applicability is as follows:

  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: This exception is supported.
  <!-- end id8 -->
  <!-- npu="A3,910b" id16 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: Not supported. The shapes of `key` and `value` must be completely the same (including Head-Dim).
  <!-- end id16 -->
- D: Head-Dim must satisfy (qD == kD && kD >= vD), and the value range is 1 to 768. The applicability of unequal D (kD > vD) is as follows:

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: Supported.
  <!-- end id9 -->
  <!-- npu="A3,910b" id17 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: Not supported. qD == kD == vD must be satisfied.
  <!-- end id17 -->
- Data shape constraints (B indicates batch size, N indicates the number of heads, S indicates sequence length, and T indicates B\*S)
  
  | Scenario | B | N | S | T |
  | --- | --- | --- | --- | --- |
  | Non-varlen | 1 to 2M (maximum 2K when prefix is used) | 1 to 256 | 1 to 1M | - |
  | Varlen | 1 to 20000 (maximum 1K when prefix is used); for <term>Ascend 950PR&950DT products</term>, 1 to 2K | 1 to 256 | 1 to 1M | 1 to 1M |

- Varlen scenarios (input_layout is TND):
  - `actual_seq_qlen` and `actual_seq_kvlen` must be passed, and their lengths must be equal and cannot be empty.
  - The `atten_mask` input does not support padding, that is, no row in `atten_mask` can be all 1s.
  - The S length of a batch in `actual_seq_qlen` can be 0, but `pse` input is not supported in this case. For example, if the actual S lengths are [2, 2, 0, 2, 2], pass `actual_seq_qlen` as [2, 4, 4, 6, 8].
  - Trailing batches can be excluded from the computation. In this case, pass the corresponding number of 0s at the end of `actual_seq_qlen` and `actual_seq_kvlen`. For example, if the actual S lengths are [2, 3, 4, 5, 6] and the last two batches do not participate in the computation, pass `actual_seq_qlen` as [2, 5, 9, 0, 0].
  - The length range of `actual_seq_qlen` and `actual_seq_kvlen` is 1 to 2K. When prefix input exists, the maximum supported length is 1K.
- The N of the input query and the N of key/value can be different, but they must be proportional, that is, Nq/Nkv must be a non-zero integer. The value range of Nq is 1 to 256. When Nq/Nkv > 1, it is GQA (grouped-query attention). When Nq/Nkv = 1, it is MHA (multi-head attention). Unless otherwise specified, N in this document indicates Nq.
- The value range of `keep_prob` is (0, 1]. When query_rope/key_rope is passed, `keep_prob` must be 1 (dropout is not supported).
- Value constraints of `sparse_mode`:
  - When `sparse_mode` is 1, 2, 3, 4, 5, 6, 7, or 8, the corresponding correct `atten_mask` must be passed; otherwise, the computation result will be incorrect.
  - When `sparse_mode` is set to 1, 2, 3, 5, or 6, the `pre_tokens` and `next_tokens` configured by the user do not take effect.
  - When `sparse_mode` is set to 0 or 4, the ranges of `atten_mask`, `pre_tokens`, and `next_tokens` must be consistent.
  - In band scenarios, `pre_tokens` and `next_tokens` must have an intersection.
  - When `sparse_mode` is set to 3, invalid row computation is automatically enabled when Sq > Skv.
  - When `sparse_mode` is set to 7 or 8, the optional parameter `pse` is not supported.
- In the `prefix` sparse computation scenario, B must not be greater than 32. Varlen scenarios do not support non-compressed prefix, that is, sparse_mode=5 is not supported.
- When `query_rope` and `key_rope` are passed:
  - They are supported only in varlen scenarios (input_layout is TND), and `atten_mask` must be passed.
  - qRoPED must be equal to kRoPED, and D must be a multiple of 8 and less than or equal to the D of query, key, and value.

  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>: The input shapes of `query_rope` and `query` differ only in the D dimension; other shape parameters must be the same. Currently, only the following scenario is supported: D of query is 128 and D of query_rope is 64.
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>: The input shapes of `key_rope` and `key` differ only in the D dimension; other shape parameters must be the same. Currently, only the following scenario is supported: D of key is 128 and D of key_rope is 64.
  <!-- end id11 -->
  - `pse` and dropout mask are not supported.
- When `sink` is passed, the shape of sink must be [head\_num].
- When `input_layout` is TND, the value range of `sparse_mode` is [0, 5) or (5, 8]. When `input_layout` has other values, the value range of `sparse_mode` is [0, 6].
- In some scenarios, if the computation amount is too large, the operator execution may time out (AI Core error type error, with errorStr: timeout or trap error). In this case, axis splitting is recommended. Note: The computation amount here is affected by B, S, N, D, and other parameters. The larger the values, the larger the computation amount.

## Examples

- Single-operator call, non-varlen scenario (BNSD)

    ```python
    import torch
    import torch_npu
    
    torch.manual_seed(0)
    BNSD = (1, 8, 16, 64)
    query = torch.randn(BNSD, dtype=torch.float16).npu()
    key = torch.randn(BNSD, dtype=torch.float16).npu()
    value = torch.randn(BNSD, dtype=torch.float16).npu()
    atten_out, softmax_max, softmax_sum, softmax_out, seed, offset, numels = torch_npu.npu_fusion_attention_v2(
        query, key, value, head_num=8, input_layout="BNSD", scale=0.088, keep_prob=0.9)
    print(atten_out.shape, atten_out.dtype)
    print(atten_out[0, 0, 0, :8])
    print(softmax_max.shape, softmax_max.dtype)
    print(softmax_max[0, 0, 0])
    print(softmax_out.shape)
    print(seed, offset, numels)
    ```

    The output is as follows:

    ```text
    torch.Size([1, 8, 16, 64]) torch.float16
    tensor([ 0.6816, -0.0999, -0.5698,  0.2695, -0.4182,  0.1345,  0.5288,  0.4639],
           dtype=torch.float16)
    torch.Size([1, 8, 16, 8]) torch.float32
    tensor([1.2028, 1.2028, 1.2028, 1.2028, 1.2028, 1.2028, 1.2028, 1.2028])
    torch.Size([0])
    0 0 2048
    ```

- Single-operator call, varlen scenario (TND)

    ```python
    import torch
    import torch_npu
    torch.manual_seed(0)
    TND = (8, 2, 64)# B=2, the lengths of the two sequences are 3 and 5, so T=8.
    query = torch.randn(TND, dtype=torch.float16).npu()
    key = torch.randn(TND, dtype=torch.float16).npu()
    value = torch.randn(TND, dtype=torch.float16).npu()
    atten_out, softmax_max, softmax_sum, softmax_out, seed, offset, numels = torch_npu.npu_fusion_attention_v2(
        query, key, value, head_num=2, input_layout="TND", scale=0.088,
        actual_seq_qlen=[3, 8], actual_seq_kvlen=[3, 8])
    print(atten_out.shape, atten_out.dtype)
    print(atten_out[:2, :, :4])
    print(softmax_max.shape)
    print(numels)
    ```

    The output is as follows:

    ```text
    torch.Size([8, 2, 64]) torch.float16
    tensor([[[ 0.2325, -1.2969, -0.8823,  0.2834],
             [ 0.5112, -0.1545,  0.9160, -0.2747]],
            [[ 0.3386, -0.9658, -0.9204,  0.3330],
             [ 0.2206, -0.4248,  0.4963,  0.6211]]], device='npu:0',
           dtype=torch.float16)
    torch.Size([8, 2, 8])
    68
    ```

- Single-operator call, passing `sink` (non-varlen scenario)

    ```python
    import torch
    import torch_npu
    
    torch.manual_seed(0)
    BNSD = (1, 8, 16, 64)
    query = torch.randn(BNSD, dtype=torch.float16).npu()
    key = torch.randn(BNSD, dtype=torch.float16).npu()
    value = torch.randn(BNSD, dtype=torch.float16).npu()
    sink = torch.randn(8, dtype=torch.float32).npu()
    atten_out, softmax_max, softmax_sum, softmax_out, seed, offset, numels = torch_npu.npu_fusion_attention_v2(
        query, key, value, head_num=8, input_layout="BNSD", scale=0.088, sink=sink)
    print(atten_out.shape)
    ```

    The output is as follows:

    ```text
    torch.Size([1, 8, 16, 64])
    ```

- Single-operator call, passing `query_rope` and `key_rope` (varlen scenario)

    ```python
    import torch
    import torch_npu
    
    torch.manual_seed(0)
    TND = (8, 2, 64)
    query = torch.randn(TND, dtype=torch.bfloat16).npu()
    query_rope = torch.randn(TND, dtype=torch.bfloat16).npu()
    key = torch.randn(TND, dtype=torch.bfloat16).npu()
    key_rope = torch.randn(TND, dtype=torch.bfloat16).npu()
    value = torch.randn(TND, dtype=torch.bfloat16).npu()
    atten_mask = torch.zeros(8, 8, dtype=torch.bool).npu()
    atten_out, softmax_max, softmax_sum, softmax_out, seed, offset, numels = torch_npu.npu_fusion_attention_v2(
        query, key, value, head_num=2, input_layout="TND", scale=0.088,
        query_rope=query_rope, key_rope=key_rope, atten_mask=atten_mask,
        actual_seq_qlen=[3, 8], actual_seq_kvlen=[3, 8])
    print(atten_out.shape, atten_out.dtype)
    ```

    The output is as follows:

    ```text
    torch.Size([8, 2, 64]) torch.bfloat16
    ```
