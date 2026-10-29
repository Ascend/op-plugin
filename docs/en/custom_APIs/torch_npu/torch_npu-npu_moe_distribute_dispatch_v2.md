# torch_npu.npu_moe_distribute_dispatch_v2

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

## Function<a name="en-us_topic_0000002203575833_section14441124184110"></a>

- **API function**: Must be used together with [torch_npu.npu_moe_distribute_combine_v2](torch_npu-npu_moe_distribute_combine_v2.md) or [torch_npu.npu_moe_distribute_combine_add_rms_norm](torch_npu-npu_moe_distribute_combine_add_rms_norm.md) to implement token dispatch_v2 and combine_v2 in parallel MoE deployment.
    - Dynamic quantization scenarios are supported. Token data is first quantized (optional), then alltoallv communication is performed in the Expert Parallelism (EP) domain, and finally allgatherv communication is performed in the Tensor Parallelism (TP) domain (optional).
    - Special expert scenarios are supported (zero experts, copy experts, and constant experts).

- Description:
    - The return value `expand_idx` of shape `(BS * K,)` in npu_moe_distribute_dispatch is replaced by `assist_info_for_combine` of shape `(A * 128,)` to carry more detailed token information and help torch_npu.npu_moe_distribute_combine_v2 efficiently perform synchronization across all ranks.
    - The input parameter `comm_alg` is added to replace the HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE environment variables.

- **Formulas**:
    - Quantization scenarios:
        - If `quant_mode` is 0 (non-quantization):

            $$
            quant\_out = x
            $$

            $$
            alltoall\_x\_out = alltoallv(quant\_out)
            $$

            $$
            expand\_x =
            \begin{cases}
            allgatherv(alltoall_x_out), & with TP communication domain \\
            alltoall_x_out, & without TP communication domain
            \end{cases}
            $$

        - If `quant_mode` is 1 (static quantization):

            $$
            quant\_out = Cast(CastToFp32(x) \times scales, dstType)
            $$

            $$
            alltoall\_x\_out = alltoallv(quant\_out)
            $$

            $$
            expand\_x =
            \begin{cases}
            allgatherv(alltoall_x_out), & with TP communication domain \\
            alltoall_x_out, & without TP communication domain
            \end{cases}
            $$

        - If `quant_mode` is 2 (dynamic quantization):

            $$
            x\_fp32 = CastToFp32(x) \times scales
            $$

            $$
            dynamic\_scales\_value = dst\_type\_max / Max(Abs(x\_fp32))
            $$

            $$
            quant\_out = Cast(x\_fp32 \times dynamic\_scales\_value, dstType)
            $$

            $$
            alltoall\_x\_out = alltoallv(quant\_out)
            $$

            $$
            alltoall\_dynamic\_scales\_out = alltoallv(1.0 / dynamic\_scales)
            $$

            $$
            expand\_x =
            \begin{cases}
            allgatherv(alltoall_x_out), & with TP communication domain \\
            alltoall_x_out, & without TP communication domain
            \end{cases}
            $$

            $$
            dynamic\_scales =
            \begin{cases}
            allgatherv(alltoall_dynamic_scales_out), & with TP communication domain \\
            alltoall_dynamic_scales_out, & without TP communication domain
            \end{cases}
            $$

        - If `quant_mode` is 3 (pergroup quantization):

            $$
            x\_fp32 = CastToFp32(x) \times scales
            $$

            $$
            dynamic\_scales\_value = dst\_type\_max / Max(Abs(x\_fp32))
            $$

            $$
            quant\_out = Cast(x\_fp32 \times dynamic\_scales\_value, dstType)
            $$

            $$
            alltoall\_x\_out = alltoallv(quant\_out)
            $$

            $$
            alltoall\_dynamic\_scales\_out = alltoallv(1.0 / dynamic\_scales)
            $$

            $$
            expand\_x =
            \begin{cases}
            \ allgatherv(alltoall\_x\_out), & \quad \text{With TP communication domain} \\
            \ alltoall\_x\_out & \quad \text{Without TP communication domain}
            \end{cases}
            $$

            $$
            dynamic\_scales =
            \begin{cases}
            allgatherv(alltoall_dynamic_scales_out), & with TP communication domain \\
            alltoall_dynamic_scales_out, & without TP communication domain
            \end{cases}
            $$

        - If `quant_mode` is 4 (mxfp quantization):

            $$
            shared\_exp = floor(log_2(max(x))) - emax
            $$

            $$
            dynamic\_scales\_value = 2^{shared\_exp}
            $$

            $$
            quant\_out = Cast(x / dynamic\_scales\_value, dstType)
            $$

            $$
            alltoall\_x\_out = alltoallv(quant\_out)
            $$

            $$
            alltoall\_dynamic\_scales\_out = alltoallv(1.0 / dynamic\_scales)
            $$

            $$
            expand\_x =
            \begin{cases}
            allgatherv(alltoall_x_out), & with TP communication domain \\
            alltoall_x_out, & without TP communication domain
            \end{cases}
            $$

            $$
            dynamic\_scales =
            \begin{cases}
            allgatherv(alltoall_dynamic_scales_out), & with TP communication domain \\
            alltoall_dynamic_scales_out, & without TP communication domain
            \end{cases}
            $$

        - If `quant_mode` is 5 (mxfp8 quantization clip implementation):

            $$
            max\_abs = max(abs(x))
            $$

            $$
            max\_abs\_clamp = max(max\_abs, 10^{-4})
            $$

            $$
            shared\_exp = ceil(log_2(max\_abs\_clamp / fp8\_max))
            $$

            $$
            dynamic\_scales\_value = 2^{shared\_exp}
            $$

            $$
            quant\_out = CastToFp8(x / dynamic\_scales\_value)
            $$

            $$
            alltoall\_x\_out = alltoallv(quant\_out)
            $$

            $$
            alltoall\_dynamic\_scales\_out = alltoallv(1.0 / dynamic\_scales)
            $$

            $$
            expand\_x =
            \begin{cases}
            \ allgatherv(alltoall\_x\_out), & \quad \text{With TP communication domain} \\
            \ alltoall\_x\_out & \quad \text{Without TP communication domain}
            \end{cases}
            $$

            $$
            dynamic\_scales =
            \begin{cases}
            \ allgatherv(alltoall\_dynamic\_scales\_out), & \quad \text{With TP communication domain} \\
            \ \ alltoall\_dynamic\_scales\_out & \quad \text{Without TP communication domain}
            \end{cases}
            $$

    - Special expert scenarios:

        Zero expert scenario (`zero_expert_num ≠ 0`):

        $$
        Moe(ori\_x)=0
        $$

        Copy expert scenario (`copy_expert_num ≠ 0`):

        $$
        Moe(ori\_x)=ori\_x
        $$

        Constant expert scenario (`const_expert_num ≠ 0`):

        $$
        Moe(ori\_x)=const\_expert\_alpha\_1*ori\_x+const\_expert\_alpha\_2*const\_expert\_v
        $$

        For details about the `ori_x`, `const_expert_alpha_1`, `const_expert_alpha_2`, and `const_expert_v` parameters, see the [torch_npu.npu_moe_distribute_combine_v2](torch_npu-npu_moe_distribute_combine_v2.md) documentation.

## Prototype<a name="en-us_topic_0000002203575833_section45077510411"></a>

```python
torch_npu.npu_moe_distribute_dispatch_v2(x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, *, scales=None, x_active_mask=None, expert_scales=None, elastic_info=None, performance_info=None, group_tp="", tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, quant_mode=0, global_bs=0, expert_token_nums_type=1, comm_alg="", zero_expert_num=0, copy_expert_num=0, const_expert_num=0, y_dtype=None, x_dtype=None, scales_dtype=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## Parameters<a name="en-us_topic_0000002203575833_section112637109429"></a>

- **`x`** (`Tensor`): Required. Token data used for computation and dispatched to other ranks based on `expert_ids`. This parameter must be 2D with shape `(BS, H)`, indicating that there are `BS` tokens. The data type can be `torch.bfloat16` or `torch.float16`. The data layout can be ND. Non-contiguous tensors are supported.

    <!-- npu="910b" id4 -->
    - <term>Atlas A2 products</term>: `torch.int32` is additionally supported. Each `torch.int32` element serves as a packing carrier for eight `torch_npu.int4` values and is passed through bitwise inside the operator.
    <!-- end id4 -->
    <!-- npu="950" id73 -->
    - <term>Ascend 950DT products</term>: additionally supports `torch_npu.hifloat8`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch_npu.float4_e2m1fn_x2`, and `torch_npu.float4_e1m2fn_x2`.
    <!-- end id73 -->

- **`expert_ids`** (`Tensor`): Required. Top-K expert indices for each token, determining the experts to which each token is dispatched. This parameter must be 2D with shape `(BS, K)`. The data type can be `torch.int32`. The data layout can be ND. Non-contiguous tensors are supported. This parameter corresponds to the `expert_ids` input of [torch_npu.npu_moe_distribute_combine_v2](torch_npu-npu_moe_distribute_combine_v2.md). The value range of elements inside the tensor is [0, moe_expert_num), and the K values within the same row must be unique.
- **`group_ep`** (`str`): Required. EP communication domain name used for expert parallelism. The string length range is [1, 128). The value must differ from `group_tp`.
- **`ep_world_size`** (`int`): Required. Size of the EP communication domain.

    <!-- npu="910b" id5 -->
    - <term>Atlas A2 products</term>: The value ranges are as follows:
        - When `comm_alg` is set to "fullmesh", the value range of `ep_world_size` is 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, or 256.
        - When `comm_alg` is set to "hierarchy", the value range of `ep_world_size` is 16, 32, or 64.
    <!-- end id5 -->
    <!-- npu="A3" id6 -->
    - <term>Atlas A3 products</term>: The supported value range is [2, 768]. When `comm_alg` is set to "hierarchy", the value range is [16, 256] and must be a multiple of 16.
    <!-- end id6 -->
    <!-- npu="950" id7 -->
    - <term>Ascend 950DT products</term>: The value range is [2, 1024].
    <!-- end id7 -->

- **`ep_rank_id`** (`int`): Required. Rank ID of the current rank within the EP communication domain. The value range is [0, `ep_world_size`). The `ep_rank_id` values of all ranks within the same EP communication domain must be unique.
- **`moe_expert_num`** (`int`): Required. Number of MoE experts. The value range is [1, 1024], and the condition `moe_expert_num % (ep_world_size - shared_expert_rank_num) = 0` must be satisfied.

    <!-- npu="A3" id8 -->
    - <term>Atlas A3 products</term>: When `comm_alg` is set to "hierarchy", the value range is [1, 512].
    <!-- end id8 -->

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`scales`** (`Tensor`): Optional. Weights of each expert. For more requirements on the shape in different products or scenarios, see [quant_mode constraints](#quant_mode_constraints).
    - This parameter must be passed when the data type of `x` is `torch.int32`, indicating the scaling parameter corresponding to each input token. This parameter must be 1D with shape `(BS,)`. The data type can be `torch.float32`. The data layout can be ND. Non-contiguous tensors are not supported. This parameter is communicated together with the corresponding tokens.

    <!-- npu="A3,910b" id9 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: This parameter is not passed in non-quantization scenarios and is optional in dynamic quantization scenarios. If provided, it must be 2D. When shared experts are present, the shape is `(shared_expert_num+moe_expert_num, H)`; when no shared experts are present, the shape is `(moe_expert_num, H)`. The data type can be `torch.float`. The data layout can be ND. Non-contiguous tensors are not supported.
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950DT products</term>: The data type can be `torch.float32` or `torch.float8_e8m0fnu`.
        - When `quant_mode` is 0, `None` or valid data can be passed. When valid data is passed, it must be a 2D tensor whose first dimension is bs and whose second dimension is less than or equal to H.
        - When `quant_mode` is 1, valid data must be passed, with a shape of `(1,)`, `(H,)`, `(shared_expert_num+moe_expert_num, H)`, or `(moe_expert_num, H)`.
        - When `quant_mode` is 2 or 3, `None` or valid data can be passed. When valid data is passed, the shape is `(shared_expert_num+moe_expert_num, H)` or `(moe_expert_num, H)`.
        - When `quant_mode` is 4 or 5, use the default value.
    <!-- end id10 -->

- **`x_active_mask`** (`Tensor`): Optional. Specifies whether a token participates in communication.

    <!-- npu="910b" id11 -->
    - <term>Atlas A2 products</term>:
        - When `comm_alg` is set to "fullmesh", this parameter must be 1D with shape `(BS,)`. The data type can be `torch.bool`. The data layout can be ND. Non-contiguous tensors are supported. The value `true` indicates that the corresponding token participates in communication, and all `true` values must precede any `false` values. For example, `{true, false, true}` is an invalid input. By default, all tokens participate in communication. When the `BS` values differ across ranks, all tokens must be valid.
        - When `comm_alg` is set to "hierarchy", it is not supported in this version. Use the default value.
    <!-- end id11 -->
    <!-- npu="950,A3" id12 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: This parameter must be 1D or 2D. When the input is 1D, the shape is `(BS,)`; when the input is 2D, the shape is `(BS, K)`. The data type can be `torch.bool`. The data layout is $ND$. Non-contiguous tensors are supported. When the input is 1D, the value `true` indicates that the corresponding token participates in communication, and all `true` values must precede any `false` values. For example, `{true, false, true}` is an invalid input. When the input is 2D, the value `true` indicates that the `expert_ids` entry corresponding to the current token participates in communication. If all K `torch.bool` values for a token are `false`, the token does not participate in communication. By default, all tokens participate in communication. When the `BS` values differ across ranks, all tokens must be valid.
    <!-- end id12 -->

- **`expert_scales`** (`Tensor`): Optional. Top-K expert weights for each token.

    <!-- npu="910b" id13 -->
    - <term>Atlas A2 products</term>: This parameter must be 2D with shape `(BS, K)`. The data type can be `torch.float`. The data layout can be ND. Non-contiguous tensors are supported.
    <!-- end id13 -->
    <!-- npu="A3" id14 -->
    - <term>Atlas A3 products</term>: This parameter is not supported currently. Use the default value.
    <!-- end id14 -->
    <!-- npu="950" id15 -->
    - <term>Ascend 950DT products</term>: `None` or a 2D tensor can be passed. When valid data is passed, the shape is `(BS, K)`. The data type can be `torch.float`. The data layout is $ND$. Non-contiguous tensors are supported.
    <!-- end id15 -->

- **`elastic_info`** (`Tensor`): Optional. Reserved parameter, not supported in the current version. Pass the default value `None`.
- **`performance_info`** (`Tensor`): Optional. Communication time for the current rank to wait for data from other ranks, in us. The communication time of each rank in a single API call is accumulated on this tensor, and the operator does not automatically clear it. Therefore, clear this tensor to zero before enabling it to record the elapsed time each time. When `None` is passed, the communication time recording function is disabled. When valid data is passed, this parameter must be 1D with shape `(ep_world_size,)`. The data type can be `torch.int64`. The data layout can be ND. Non-contiguous tensors are supported.
- **`group_tp`** (`str`): Optional. TP communication domain name, that is, the communication domain for data parallelism. This parameter is required if TP domain communication exists. If there is no TP domain communication, use the default value "".

    <!-- npu="910b" id16 -->
    - <term>Atlas A2 products</term>: In eager mode, use the default value. In graph mode, the value must be identical to that of `group_ep`.
    <!-- end id16 -->
    <!-- npu="A3" id17 -->
    - <term>Atlas A3 products</term>: The string length range is [1, 128), and the value must differ from `group_ep`.
    <!-- end id17 -->
    <!-- npu="950" id18 -->
    - <term>Ascend 950DT products</term>: TP domain communication is not supported. Use the default input.
    <!-- end id18 -->

- **`tp_world_size`** (`int`): Optional. Size of the TP communication domain. This parameter is required only when TP domain communication is involved.

    <!-- npu="910b" id19 -->
    - <term>Atlas A2 products</term>: TP domain communication is not supported. Use the default value `0`.
    <!-- end id19 -->
    <!-- npu="A3" id20 -->
    - <term>Atlas A3 products</term>: When TP domain communication is involved, the value range is [0, 2]. The values `0` and `1` indicate that there is no TP domain communication, and the value `2` indicates that there is TP domain communication.
    <!-- end id20 -->
    <!-- npu="950" id21 -->
    - <term>Ascend 950DT products</term>: TP domain communication is not supported. Use the default value `0`.
    <!-- end id21 -->

- **`tp_rank_id`** (`int`): Optional. Rank ID of the current rank within the TP communication domain. This parameter is required only when TP domain communication is involved.

    <!-- npu="910b" id22 -->
    - <term>Atlas A2 products</term>: Currently, this parameter is not supported. Retain the default value.
    <!-- end id22 -->
    <!-- npu="A3" id23 -->
    - <term>Atlas A3 products</term>: When TP domain communication is involved, the value range is [0, 1]. The default value is `0`, and the `tp_rank_id` values of all ranks within the same TP communication domain must be unique. If there is no TP domain communication, pass `0`.
    <!-- end id23 -->
    <!-- npu="950" id24 -->
    - <term>Ascend 950DT products</term>: TP domain communication is not supported. Use the default value `0`.
    <!-- end id24 -->

- **`expert_shard_type`** (`int`): Optional. Layout type of shared-expert ranks.

    <!-- npu="910b" id25 -->
    - <term>Atlas A2 products</term>: Currently, this parameter is not supported. Retain the default value.
    <!-- end id25 -->
    <!-- npu="950,A3" id26 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: Currently only 0 is supported, indicating that shared-expert ranks are placed before MoE expert ranks.
    <!-- end id26 -->

- **`shared_expert_num`** (`int`): Optional. Number of shared experts, where a shared expert can be replicated and deployed across multiple ranks.

    <!-- npu="910b" id27 -->
    - <term>Atlas A2 products</term>: Expert sharing is not supported. Retain the default value.
    <!-- end id27 -->
    <!-- npu="950,A3" id28 -->
    - <term>Atlas A3 products</term> and <term>Ascend 950DT products</term>: The value range is [0, 4]. A value of `0` indicates that there is no shared expert, and the default value is 1.
    <!-- end id28 -->

- **`shared_expert_rank_num`** (`int`): Optional. Number of shared-expert ranks.

    <!-- npu="910b" id29 -->
    - <term>Atlas A2 products</term>: Shared experts are not supported. Use the default value.
    <!-- end id29 -->
    <!-- npu="950,A3" id30 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The value range is [0, ep_world_size). A value of 0 indicates that there is no shared expert; a non-zero value requires `shared_expert_rank_num % shared_expert_num = 0`.
    <!-- end id30 -->

- **`quant_mode`** (`int`): Optional. Quantization mode.

    <!-- npu="A3,910b" id31 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: 0 indicates non-quantization (default), and 2 indicates dynamic quantization. When `quant_mode` is 2, `dynamic_scales` is not `None`; when `quant_mode` is 0, `dynamic_scales` is `None`. When the data type of `x` is `torch.int32`, only the value 0 is supported, and `dynamic_scales` is not `None`.
    <!-- end id31 -->
    <!-- npu="950" id32 -->
    - <term>Ascend 950DT products</term>: 0 indicates non-quantization (default), 1 indicates static quantization, 2 indicates dynamic per-token quantization, 3 indicates pergroup quantization, 4 indicates mxfp quantization, and 5 indicates the mxfp quantization clip implementation. When `quant_mode` is 2, 3, 4, or 5, `dynamic_scales` is not `None`; when `quant_mode` is 0 or 1, `dynamic_scales` is `None`.
    <!-- end id32 -->

- **`global_bs`** (`int`): Optional. Global batch size of the EP domain. When the BS values of each rank are different, `max_bs*ep_world_size` can be passed, where `max_bs` indicates the maximum BS of a single rank. When the BS values of each rank are the same, the value can be 0 or `BS*ep_world_size`.
- **`expert_token_nums_type`** (`int`): Optional. Value type of the output `expert_token_nums`. The value range is [0, 1]. `0` indicates the prefix sum of the numbers of tokens received by each expert, and `1` indicates the number of tokens received by each expert (default).
- **`comm_alg`** (`str`): Optional. Communication affinity memory layout algorithm.

    <!-- npu="910b" id33 -->
    - <term>Atlas A2 products</term>: In the current version, "", "fullmesh", and "hierarchy" are supported. Configuring "hierarchy" with driver 25.0.RC1.1 or later is recommended.
        - "": When HCCL_INTRA_PCIE_ENABLE=1 and HCCL_INTRA_ROCE_ENABLE=0 are configured, the "hierarchy" algorithm is used; otherwise, the "fullmesh" algorithm is used. This mode is not recommended.
        - "fullmesh": Token data is sent directly through RDMA to the ranks where the top-K target experts reside.
        - "hierarchy": Token data is sent twice, first across servers and then within a server. RDMA communication is used only between the same-numbered ranks on different servers, and HCCS communication is used within a server.
    <!-- end id33 -->
    <!-- npu="950,A3" id34 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: In the current version, only three input modes are supported: "", "fullmesh_v1", and "fullmesh_v2".
        - "": The default value, which does not enable the fullmesh_v2 template.
        - "fullmesh_v1": The fullmesh_v2 template is not enabled.
        - "fullmesh_v2": The fullmesh_v2 template is enabled. `comm_alg` takes effect only when `tp_world_size` is 1, and it cannot be enabled when the BS values of ranks are inconsistent, `x_active_mask` is used, or special expert scenarios are involved.
    <!-- end id34 -->

- **`zero_expert_num`** (`int`): Optional. Number of zero experts.

    <!-- npu="910b" id35 -->
    - <term>Atlas A2 products</term>: Not supported in the current version; pass `0`.
    <!-- end id35 -->
    <!-- npu="950,A3" id36 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The value range is [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid zero-expert IDs are in the range [moe_expert_num, moe_expert_num+zero_expert_num).
    <!-- end id36 -->

- **`copy_expert_num`** (`int`): Optional. Number of copy experts.

    <!-- npu="910b" id37 -->
    - <term>Atlas A2 products</term>: This parameter is currently not supported. Set it to `0`.
    <!-- end id37 -->
    <!-- npu="950,A3" id38 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The value range is [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid copy-expert IDs are in the range [moe_expert_num+zero_expert_num, moe_expert_num+zero_expert_num+copy_expert_num).
    <!-- end id38 -->

- **`const_expert_num`** (`int`): Optional. Number of constant experts.

    <!-- npu="910b" id39 -->
    - <term>Atlas A2 products</term>: Not supported in the current version; pass `0`.
    <!-- end id39 -->
    <!-- npu="950,A3" id40 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The value range is [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid constant-expert IDs are in the range [moe_expert_num+zero_expert_num+copy_expert_num, moe_expert_num+zero_expert_num+copy_expert_num+const_expert_num).
    <!-- end id40 -->

- **`y_dtype`** (`int`): Optional. Data type of the output.

    <!-- npu="A3,910b" id41 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: This parameter is not supported currently.
    <!-- end id41 -->
    <!-- npu="950" id42 -->
    - <term>Ascend 950DT products</term>: This parameter should be specified when `quant_mode` is not 0. If not specified, the default type is `torch.int8`.
    <!-- end id42 -->

- **`x_dtype`** (`int`): Optional. Data type of the input tensor `x`. If it is a data type not natively supported by PyTorch, it must be specified through this parameter.
- **`scales_dtype`** (`int`): Optional. Data type of the input tensor `scales`. If it is a data type not natively supported by PyTorch, it must be specified through this parameter.

## Return Values

- **`expand_x`** (`Tensor`): Token data received by the current rank. This parameter must be 2D with shape `(max(tp_world_size, 1) * A, H)`, where A indicates the maximum number of tokens that can be received in the EP communication domain. The data layout can be ND. Non-contiguous tensors are supported. In quantization scenarios, the type is `torch.int8`; in non-quantization scenarios, the type is the same as that of `x`.

    <!-- npu="A3,910b" id43 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data type can be `torch.bfloat16`, `torch.float16`, or `torch.int8`.
    <!-- end id43 -->
    <!-- npu="910b" id44 -->
    - <term>Atlas A2 products</term>: `torch.int32` is additionally supported in non-quantization scenarios.
    <!-- end id44 -->
    <!-- npu="950" id45 -->
    - <term>Ascend 950DT products</term>: The data type can be `torch.float16`, `torch.bfloat16`, `torch.int8`, `torch.float8_e5m2`, `torch.float8_e4m3fn`, `torch_npu.hifloat8`, `torch_npu.float4_e2m1fn_x2`, or `torch_npu.float4_e1m2fn_x2`.
    <!-- end id45 -->

- **`dynamic_scales`** (`Tensor`): Dynamically quantized parameters obtained through computation, or scaling parameters communicated together with tokens whose data type is `torch.int32`. This output is available only when `quant_mode` is not 0 or 1. When the data type of `x` is `torch.int32` and `quant_mode` is 0, this output is available, and the first several elements corresponding to the actually received token count are valid. The data layout can be ND. Non-contiguous tensors are supported.

    <!-- npu="A3,910b" id46 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: This parameter must be 1D with shape `(A,)`. The data type can be `torch.float`.
    <!-- end id46 -->
    <!-- npu="950" id47 -->
    - <term>Ascend 950DT products</term>: The data type can be `torch.float32` or `torch.float8_e8m0fnu`. For details about the shape requirements in different scenarios, see [quant_mode constraints](#quant_mode_constraints):
        - When `quant_mode` is 0, if the input `x` type is `torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`/`torch_npu.hifloat8` and the `expand_x` type is the same as the input `x` type, this parameter must be a 2D tensor whose first dimension is A; otherwise, no valid value is output.
        - When `quant_mode` is 2, the shape is `(A,)`; when `quant_mode` is 3, the shape is `(A, ceil(H/128))`.
        - When `quant_mode` is 4 or 5, the shape is `(A, ceil(H/32))` and ceil(H/32) % 2 == 0.
    <!-- end id47 -->

- **`assist_info_for_combine`** (`Tensor`): Number of tokens dispatched to the same expert. This parameter must be 1D with shape `(A * 128,)`. The data type can be `torch.int32`. The data layout can be ND. Non-contiguous tensors are supported. This parameter corresponds to the `assist_info_for_combine` input of [torch_npu.npu_moe_distribute_combine_v2](torch_npu-npu_moe_distribute_combine_v2.md).
- **`expert_token_nums`** (`Tensor`): Number of tokens actually received by each expert on the current rank. This parameter must be 1D with shape `(local_expert_num,)`. The data type can be `torch.int64`. The data layout is ND. Non-contiguous tensors are supported.
- **`ep_recv_counts`** (`Tensor`): Number of tokens (represented as a prefix sum) received by each rank in the EP communication domain. This parameter must be 1D. The data type can be `torch.int32`. The data layout can be ND. Non-contiguous tensors are supported. This parameter corresponds to the `ep_send_counts` input of [torch_npu.npu_moe_distribute_combine_v2](torch_npu-npu_moe_distribute_combine_v2.md).

    <!-- npu="910b" id48 -->
    - <term>Atlas A2 products</term>: The shape must be `(moe_expert_num+2*global_bs*K*server_num,)`. The first `moe_expert_num` elements indicate the token counts received by each expert on the current rank from other ranks within the EP communication domain, represented as prefix sums. The `2*global_bs*K*server_num` elements are used to store the token counts that can be reduced in advance during the combine operation and the communication buffer offsets before inter-server and intra-server communication. When `global_bs` is `0`, the value is calculated as `bs*ep_world_size`.
    <!-- end id48 -->
    <!-- npu="950,A3" id49 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The shape must be (ep_world_size*max(tp_world_size, 1)*local_expert_num,).
    <!-- end id49 -->

- **`tp_recv_counts`** (`Tensor`): Number of tokens received by each rank in the TP communication domain. This parameter corresponds to the `tp_send_counts` input of [torch_npu.npu_moe_distribute_combine_v2](torch_npu-npu_moe_distribute_combine_v2.md).

    <!-- npu="950,910b" id50 -->
    - <term>Atlas A2 products</term>, <term>Ascend 950DT products</term>: The TP communication domain is not supported, and this output is not available.
    <!-- end id50 -->
    <!-- npu="A3" id51 -->
    - <term>Atlas A3 products</term>: The TP communication domain is supported. This parameter must be 1D with shape `(tp_world_size,)`. The data type can be `torch.int32`. The data layout is ND. Non-contiguous tensors are supported.
    <!-- end id51 -->

- **`expand_scales`** (`Tensor`): Output after the alltoallv operation is performed on `expert_scales` and `x`.

    <!-- npu="910b" id52 -->
    - <term>Atlas A2 products</term>: This parameter must be 1D with shape `(A,)`. The data type can be `torch.float`. The data layout is ND. Non-contiguous tensors are supported.
    <!-- end id52 -->
    <!-- npu="A3" id53 -->
    - <term>Atlas A3 products</term>: This output is not supported. `None` is returned.
    <!-- end id53 -->
    <!-- npu="950" id54 -->
    - <term>Ascend 950DT products</term>: The output is available only when valid data is passed for `expert_scales`. This parameter must be 1D with shape `(A,)`. The data type can be `torch.float`. The data layout is $ND$. Non-contiguous tensors are supported.
    <!-- end id54 -->

## Constraints

- Atlas A2 training products/Atlas A2 inference products: The value range of `BS` is as follows:
<!-- npu="950" id55 -->
- When `comm_alg` is set to `"fullmesh"`: 0 < `BS` <= 256.
    - <term>Ascend 950DT products</term>: Only UB Memory communication is supported.
<!-- end id55 -->

- This API supports single-operator mode and static graph mode, and must be used together with `npu_moe_distribute_combine_v2` or `npu_moe_distribute_combine_add_rms_norm`.
- Element values inside the `assist_info_for_combine`, `ep_recv_counts`, `tp_recv_counts`, and `expand_scales` tensors output by `npu_moe_distribute_dispatch_v2` can vary across different product models, communication algorithms, or software versions. When using this API, pass these tensors directly to the corresponding parameters of `npu_moe_distribute_combine_v2`. Other service logic of the model must not depend on them.
- The values of the `group_ep`, `ep_world_size`, `moe_expert_num`, `group_tp`, `tp_world_size`, `expert_shard_type`, `shared_expert_num`, `shared_expert_rank_num`, and `global_bs` parameters must remain identical across all ranks during the API call process. In addition, the values of `group_ep`, `ep_world_size`, `group_tp`, `tp_world_size`, `expert_shard_type`, and `global_bs` must also remain identical across different layers in the network and must match the corresponding parameters in [torch_npu.npu_moe_distribute_combine_v2](torch_npu-npu_moe_distribute_combine_v2.md).

<!-- npu="A3" id56 -->
- <term>Atlas A3 products</term>: The value range of `BS` is as follows.
<!-- end id56 -->
- moe\_expert\_num + zero\_expert\_num + copy\_expert\_num + const\_expert\_num < MAX\_INT32.
- Variables used in the shapes of parameters are as follows:
    - `K`: Number of top-K experts selected. The value range is 0 < `K` <= 16, and the condition 0 < `K` <= `moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num` must be satisfied. When `comm_alg` is set to `""` or `"fullmesh_v2"`, the value range is 0 < `K` <= 12.
        - For shared experts, `A=BS*ep_world_size*shared_expert_num/shared_expert_rank_num` must be satisfied.
        - For MoE experts, when `global_bs` is `0`, `A>=BS*ep_world_size*min(local_expert_num, K)` must be satisfied; when `global_bs` is not `0`, `A>=global_bs*min(local_expert_num, K)` must be satisfied.

    - `server_num`: Number of server nodes. Valid values are `2`, `4`, or `8`.

        <!-- npu="910b" id57 -->
        - <term>Atlas A2 products</term>: This variable is used only in shapes for this scenario.
            - When the data type of `x` is `torch.int32`, only `comm_alg` set to "fullmesh" is supported, and the value range of `H` is (0, 5120], which must be a multiple of 32.
            - When `comm_alg` is set to "fullmesh" and the data type of `x` is not `torch.int32`, the value range of `H` is (0, 7168], which must be a multiple of 32.
            - When `comm_alg` is set to "hierarchy" and the driver version is not earlier than 25.0.RC1.1, the value range of `H` is (0, 10 * 1024], which must be a multiple of 32.
        <!-- end id57 -->
        <!-- npu="A3" id58 -->
        - <term>Atlas A3 products</term>: The value is in the range [1024, 8192].
        <!-- end id58 -->
        <!-- npu="950" id59 -->
        - <term>Ascend 950DT products</term>: The value range is [1024, 8192], and when `expand_x` is of the `torch_npu.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2` type, the value must be an even number.
        <!-- end id59 -->

    - BS: batch sequence size, that is, the number of tokens finally output by the current rank.

        <!-- npu="910b" id60 -->
        - <term>Atlas A2 products</term>: The value range is 0 < BS ≤ 256.
        <!-- end id60 -->
        <!-- npu="A3" id61 -->
        - <term>Atlas A3 products</term>: The value range is 0 < BS ≤ 512.
        <!-- end id61 -->
        <!-- npu="950" id62 -->
        - <term>Ascend 950DT products</term>: The value range is 0 < BS <= 512.
        <!-- end id62 -->

    - K: number of top-K experts selected. When `comm_alg` is set to "" or "fullmesh_v1", the value range is 0 < K <= 16; when `comm_alg` is set to "fullmesh_v2", 0 < K <= 12. In addition, 0 < K <= moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num must be satisfied.
    - server_num: number of server nodes. Only the values 2, 4, and 8 are supported.

        <!-- npu="910b" id63 -->
        - <term>Atlas A2 products</term>: This variable is used only in the shapes of this scenario.
        <!-- end id63 -->

    - local_expert_num: Number of experts on the current rank.
        - For shared-expert ranks, `local_expert_num` is 1.
        - For MoE expert ranks, `local_expert_num = moe_expert_num/(ep_world_size - shared_expert_rank_num)`. When `local_expert_num` is greater than 1, communication in the TP domain is not supported.

        <!-- npu="950,A3" id64 -->
        - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The condition `0 < local_expert_num * ep_world_size <= 2048` must be satisfied.
        <!-- end id64 -->

- The constraints related to quant_mode are as follows:
    <a id="quant_mode_constraints"></a>

    <!-- npu="A3,910b" id65 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>

        **Table 1** quant_mode constraints

        | quant_mode | Quantization Mode | expand_x Parameter | scales Parameter | dynamic_scales |
        | --- | --- | --- | --- | --- |
        | 0 | Non-quantization | `torch.float16`/`torch.bfloat16` | None | No valid value is output. |
        | 2 | Dynamic quantization | `torch.int8` | `None` or a non-null value is supported. When non-null, the shape is (sharedExpertNum+moeExpertNum, H) or (moeExpertNum, H), and the data type is `torch.float32`. | The shape is (A, ). The data type is `torch.float32`. |
    <!-- end id65 -->
    <!-- npu="950" id66 -->
    - <term>Ascend 950DT products</term>

        **Table 2** quant_mode constraints

        | quant_mode | Quantization Mode | expand_x Parameter | scales Parameter | dynamic_scales |
        | --- | --- | --- | --- | --- |
        | 0 | Non-quantization | `torch.float16`/`torch.bfloat16`/`torch_npu.hifloat8`/`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2` | If the expand_x type is `torch.float16`/`torch.bfloat16`, pass None.<br>If the expand_x type is `torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`, the input must be a 2D tensor whose first dimension is bs and whose second dimension is less than H, and the data type is `torch.float8_e8m0fnu`.<br>If the expand_x type is `torch_npu.hifloat8` and is the same as the input x, the input must be a 2D tensor whose first dimension is bs, and the data type is `torch.float32`; otherwise, pass None. | If the expand_x type is `torch.float16`/`torch.bfloat16`, no valid value is output.<br>If the expand_x type is `torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`, a 2D tensor is output whose first dimension is A and whose second dimension and data type are the same as those of the input scales.<br>If the expand_x type is `torch_npu.hifloat8` and is the same as the input x, a 2D tensor is output whose first dimension is A and whose second dimension and data type are the same as those of the input scales; otherwise, no valid value is output. |
        | 1 | Static quantization | `torch.int8` | Not null. The data type is `torch.float32`.<br>If scales represents the quantization coefficient, the shape is (1, ).<br>If scales represents the smooth weight shared by each expert, the shape is (H, ).<br>If scales represents the quantization coefficient with the smooth weight shared by experts, the shape is (sharedExpertNum+moeExpertNum, H) when shared-expert ranks are present, and (moeExpertNum, H) when no shared-expert ranks are present. | No valid value is output. |
        | 1 | Static quantization | `torch_npu.hifloat8` | Not null. The shape is (1, ), and the data type is `torch.float32`. | No valid value is output. |
        | 2 | Per-token dynamic quantization | `torch.int8`/`torch.float8_e4m3fn`/`torch.float8_e5m2` | `None` or a non-null value is supported. When non-null, the shape is (sharedExpertNum+moeExpertNum, H) when shared-expert ranks are present and (moeExpertNum, H) when no shared-expert ranks are present, and the data type is `torch.float32`. | A valid value is output. The shape is (A, ). The data type is `torch.float32`. |
        | 3 | Per-group dynamic quantization | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `None` or a non-null value is supported. When non-null, the shape is (sharedExpertNum+moeExpertNum, H) when shared-expert ranks are present and (moeExpertNum, H) when no shared-expert ranks are present, and the data type is `torch.float32`. | A valid value is output. The shape is (A, Ceil(H, 128)). The data type is `torch.float32`. |
        | 4 | mx quantization | `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2` | None. | A valid value is output. The shape is (A, Ceil(H, 32)) and ceil(H/32) % 2 == 0. The data type is `torch.float8_e8m0fnu`. |
        | 5 | mxfp8 quantization clip implementation | `torch.float8_e4m3fn`/`torch.float8_e5m2` | None. | A valid value is output. The shape is (A, Ceil(H, 32)) and ceil(H/32) % 2 == 0. The data type is `torch.float8_e8m0fnu`. |
    <!-- end id66 -->

- HCCL communication domain buffer size:

    Before calling this API, verify that the configured HCCL communication domain buffer size is reasonable. The unit is MB, and the default value is 200 MB if not configured.

    <!-- npu="910b" id67 -->
    - <term>Atlas A2 products</term>: In this scenario, configuration through the environment variable `HCCL_BUFFSIZE` is supported.
        - When `comm_alg` is set to "": The HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE environment variables take effect only in this configuration. The "fullmesh" or "hierarchy" formula is selected based on the HCCL_INTRA_PCIE_ENABLE and HCCL_INTRA_ROCE_ENABLE configurations.
        - When `comm_alg` is set to "fullmesh" and the data type of `x` is `torch.int32`: The configured size must be greater than or equal to `2 * (BS * ep_world_size * min(local_expert_num, K) * (H * sizeof(`torch.int32`) + 32B) + 2MB)`.
        - When `comm_alg` is set to "fullmesh" and the data type of `x` is not `torch.int32`: The configured size must be greater than or equal to `2 * (BS * ep_world_size * min(local_expert_num, K) * H * sizeof(`torch.uint16`) + 2MB)`.
        - When `comm_alg` is set to "hierarchy": The configured size must be equal to `moe_expert_num * BS * (H * sizeof(dtype_x) + 4 * ((K + 7)/8 * 8) * sizeof(`torch.uint32`)) + 4MB + 100MB`, and the condition `moe_expert_num/(ep_world_size - shared_expert_rank_num) <= 24` is not required.
    <!-- end id67 -->
    <!-- npu="A3" id68 -->
    - <term>Atlas A3 products</term>: In this scenario, the parameter can be configured not only through the environment variable HCCL\_BUFFSIZE but also through hccl\_buffer\_size (see the hccl\_buffer\_size section under Performance Tuning > Performance Tuning Methods > Communication Optimization > Optimization Methods in [PyTorch Training Model Migration and Tuning](https://hiascend.com/document/redirect/canncommercial-ptmigr)).
        - Within the EP communication domain, when `comm_alg` is set to "fullmesh_v1" or "": The configured size must be greater than or equal to `2 * (local_expert_num * max_bs * ep_world_size * Align512(Align32(2 * H) + 64) + (K + shared_expert_num) * max_bs * Align512(2 * H))`.
        - Within the EP communication domain, when `comm_alg` is set to "fullmesh_v2": The configured size must be greater than or equal to `2 * (local_expert_num * max_bs * ep_world_size * 480Align512(Align32(2 * H) + 64) + (K + shared_expert_num) * max_bs * Align512(2 * H))`.
        - Within the TP communication domain: The configured size must be greater than or equal to `(A * Align512(Align32(H * 2) + 44) + A * Align512(H * 2)) * 2`.
        - The alignment functions are defined as follows: 480Align512(x) = ((x+480-1)/480) \* 512, Align512(x) = ((x+512-1)/512) \* 512, and Align32(x) = ((x+32-1)/32) \* 32.
    <!-- end id68 -->
    <!-- npu="950" id69 -->
    - <term>Ascend 950DT products</term>: The value range is as follows:
        - Within the EP communication domain, when `comm_alg` is set to "fullmesh_v1" or "": The configured size must be greater than or equal to `2 * (local_expert_num * max_bs * ep_world_size * Align512(Align32(2 * H) + 64) + (K + shared_expert_num) * max_bs * Align512(2 * H)) + 1024 * 1024 * 10`.
        - Within the EP communication domain, when `comm_alg` is set to "fullmesh_v2": The configured size must be greater than or equal to `2 * (local_expert_num * max_bs * ep_world_size * 480Align512(Align32(2 * H) + 64) + (K + shared_expert_num) * max_bs * Align512(2 * H)) + 1024 * 1024 * 10`.
        - The alignment functions are defined as follows: 480Align512(x) = ((x+480-1)/480) \* 512, Align512(x) = ((x+512-1)/512) \* 512, and Align32(x) = ((x+32-1)/32) \* 32.
    <!-- end id69 -->

<!-- npu="910b" id70 -->
- <term>Atlas A2 products</term>: The `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE` environment variables are no longer recommended. You are advised to set `comm_alg` to "hierarchy".
<!-- end id70 -->
- Communication domain usage constraints:
    - The `npu_moe_distribute_dispatch_v2` and `npu_moe_distribute_combine_v2` operators within a single model must operate in the same EP communication domain, which must not include other operators.
    - The `npu_moe_distribute_dispatch_v2` and `npu_moe_distribute_combine_v2` operators within a single model must either operate in the same TP communication domain or both operate without a TP communication domain. When a TP communication domain is involved, this domain must not include other operators.

    <!-- npu="A3" id71 -->
    - <term>Atlas A3 products</term>: Nodes in a communication domain must reside within the same SuperPoD. Cross-SuperPoD deployment is not supported.
    <!-- end id71 -->

<!-- npu="910b" id72 -->
- Networking constraints:
    - <term>Atlas A2 products</term>: In multi-server deployments, only switch-based networking is supported. Direct connections between two servers are not supported.
<!-- end id72 -->

- Version mapping constraints:

    In static graph mode, starting with TorchNPU 8.0.0, the TorchNPU framework performs strict validation between the Meta inference results and inferShape inference results of the output of the last node in the static graph. If the graph contains only one Dispatch_v2 operator and the CANN version is earlier than the TorchNPU version, a shape mismatch error may occur. You are advised to upgrade the CANN version. For details about version compatibility information, see section **Related Product Versions** in the [TorchNPU Release Notes](https://gitcode.com/Ascend/pytorch/blob/master/docs/en/release_notes.md).

- In formulas in this document, the division sign (/) indicates integer division.

## Examples<a name="en-us_topic_0000002203575833_section14459801435"></a>

- Single-operator call

    ```python
    import os
    import torch
    import random
    import torch_npu
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp

    # Control mode
    quant_mode = 2  # 2 indicates dynamic quantization
    is_dispatch_scales = True  # For dynamic quantization, you can choose whether to pass scales
    input_dtype = torch.bfloat16  # Output data type
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # Number of dies per host
    shared_expert_rank_num = 0  # Number of shared experts
    moe_expert_num = 32  # Number of MoE experts
    bs = 8  # Number of tokens
    h = 7168  # Length of each token
    k = 8
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - shared_expert_rank_num
    local_moe_expert_num = moe_expert_num // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)
    zero_expert_num = 1
    copy_expert_num = 1
    const_expert_num = 1

    def gen_const_expert_alpha_1():
        const_expert_alpha_1 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1

    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_2

    def gen_const_expert_v():
        const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_v

    def get_new_group(rank):
        for i in range(tp_world_size):
            # Result when tp_world_size = 2 and ep_world_size = 8: [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
            ep_ranks = [x * tp_world_size + i for x in range(ep_world_size)]
            ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group_t = ep_group
                print(f"rank:{rank} ep_ranks:{ep_ranks}")
        for i in range(ep_world_size):
            # Result when tp_world_size = 2 and ep_world_size = 8: [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
            tp_ranks = [x + tp_world_size * i for x in range(tp_world_size)]
            tp_group = dist.new_group(backend="hccl", ranks=tp_ranks)
            if rank in tp_ranks:
                tp_group_t = tp_group
                print(f"rank:{rank} tp_ranks:{tp_ranks}")
        return ep_group_t, tp_group_t

    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info

    def warm_up_dispatch(rank, group_ep, group_tp):
        x_warm_up = torch.empty(size=[1, h], dtype=input_dtype).uniform_(-1024, 1024).to(input_dtype).npu()
        expert_ids_warm_up = torch.arange(0, k, dtype=torch.int32).unsqueeze(0).npu()
        dispatch_kwargs_before = get_dispatch_kwargs_warmup(
            x_warm_up=x_warm_up,
            expert_ids_warm_up=expert_ids_warm_up,
            group_ep=group_ep,
            group_tp=group_tp,
            ep_rank_id=rank // tp_world_size,
            tp_rank_id=rank % tp_world_size,
        )
        (
            expand_x, dynamic_scales, expand_idx,
            expert_token_nums, ep_recv_counts, tp_recv_counts, _
        ) = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs_before)
        return expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts

    def get_dispatch_kwargs_warmup(
        x_warm_up, expert_ids_warm_up, group_ep, group_tp, ep_rank_id, tp_rank_id,
    ):
        x_warm_up = x_warm_up.to(input_dtype).npu()
        expert_ids_warm_up = expert_ids_warm_up.to(torch.int32).npu()
        return {
            'x': x_warm_up,
            'expert_ids': expert_ids_warm_up,
            'x_active_mask': None,
            'group_ep': group_ep,
            'group_tp': group_tp,
            'ep_rank_id': ep_rank_id,
            'tp_rank_id': tp_rank_id,
            'ep_world_size': ep_world_size,
            'tp_world_size': tp_world_size,
            'expert_shard_type': 0,
            'shared_expert_num': 0,
            'shared_expert_rank_num': shared_expert_rank_num,
            'moe_expert_num': moe_expert_num,
            'scales': None,
            'quant_mode': 2,
            'global_bs': 16,
        }

    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        ep_group, tp_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        tp_hcomm_info = get_hcomm_info(rank, tp_group)

        # Create input tensors
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = torch.tensor([[5, 7, 17, 4, 2, 6, 11, 16],
                                [10, 12, 13, 15, 19, 4, 18, 1],
                                [19, 33, 1, 17, 9, 5, 0, 32],
                                [19, 11, 17, 0, 10, 5, 7, 9],
                                [10, 16, 11, 17, 33, 8, 9, 3],
                                [12, 19, 5, 7, 1, 3, 18, 16],
                                [11, 9, 13, 16, 12, 33, 17, 14],
                                [16, 4, 9, 5, 0, 10, 11, 17]], dtype=torch.int32).npu()
        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()

        scales_shape = (1 + moe_expert_num, h) if shared_expert_rank_num else (moe_expert_num, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None

        const_expert_alpha_1 = gen_const_expert_alpha_1().npu()
        const_expert_alpha_2 = gen_const_expert_alpha_2().npu()
        const_expert_v = gen_const_expert_v().npu()

        out = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)

        expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=expert_ids,
            group_ep=ep_hcomm_info,
            group_tp=tp_hcomm_info,
            ep_world_size=ep_world_size,
            tp_world_size=tp_world_size,
            ep_rank_id=rank // tp_world_size,
            tp_rank_id=rank % tp_world_size,
            expert_shard_type=0,
            shared_expert_rank_num=shared_expert_rank_num,
            moe_expert_num=moe_expert_num,
            scales=scales,
            quant_mode=quant_mode,
            global_bs=globalBS,
            zero_expert_num=zero_expert_num,
            copy_expert_num=copy_expert_num,
            const_expert_num=const_expert_num)

        if is_quant:
            expand_x = expand_x.to(input_dtype)

        x = torch_npu.npu_moe_distribute_combine_v2(expand_x=expand_x,
                                                expert_ids=expert_ids,
                                                assist_info_for_combine=assist_info_for_combine,
                                                ep_send_counts=ep_recv_counts,
                                                tp_send_counts=tp_recv_counts,
                                                expert_scales=expert_scales,
                                                group_ep=ep_hcomm_info,
                                                group_tp=tp_hcomm_info,
                                                ep_world_size=ep_world_size,
                                                tp_world_size=tp_world_size,
                                                ep_rank_id=rank // tp_world_size,
                                                tp_rank_id=rank % tp_world_size,
                                                expert_shard_type=0,
                                                shared_expert_rank_num=shared_expert_rank_num,
                                                moe_expert_num=moe_expert_num,
                                                global_bs=globalBS,
                                                ori_x=x,
                                                const_expert_alpha_1=const_expert_alpha_1,
                                                const_expert_alpha_2=const_expert_alpha_2,
                                                const_expert_v=const_expert_v,
                                                zero_expert_num=zero_expert_num,
                                                copy_expert_num=copy_expert_num,
                                                const_expert_num=const_expert_num)
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')

    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"global_bs={globalBS}")
        print(f"shared_expert_rank_num={shared_expert_rank_num}")
        print(f"moe_expert_num={moe_expert_num}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
        print(f"tp_world_size={tp_world_size}", flush=True)
        print(f"ep_world_size={ep_world_size}", flush=True)

        if tp_world_size != 1 and local_moe_expert_num > 1:
            print("unSupported tp = 2 and local moe > 1")
            exit(0)
        if shared_expert_rank_num > ep_world_size:
            print("shared_expert_rank_num cannot be greater than ep_world_size")
            exit(0)
        if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
            print("ep_world_size must be an integer multiple of shared_expert_rank_num")
            exit(0)
        if moe_expert_num % moe_rank_num != 0:
            print("moe_expert_num must be an integer multiple of moe_rank_num")
            exit(0)
        p_list = []
        for rank in range(rank_per_dev):
            p = Process(target=run_npu_process, args=(rank,))
            p_list.append(p)

        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

        print("run npu success.")
    ```

- Graph mode call

    ```python
    # Only static graphs are supported
    import os
    import torch
    import random
    import torch_npu
    import torchair
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import time

    # Control mode
    quant_mode = 2  # 2 indicates dynamic quantization
    is_dispatch_scales = True  # For dynamic quantization, you can choose whether to pass scales
    input_dtype = torch.bfloat16  # Output data type
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # Number of dies per host
    shared_expert_rank_num = 0  # Number of shared experts
    moe_expert_num = 32  # Number of MoE experts
    bs = 8  # Number of tokens
    h = 7168  # Length of each token
    k = 8
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - shared_expert_rank_num
    local_moe_expert_num = moe_expert_num // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)

    zero_expert_num = 1
    copy_expert_num = 1
    const_expert_num = 1

    class MOE_DISTRIBUTE_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, expert_ids, group_ep, group_tp, ep_world_size, tp_world_size,
                    ep_rank_id, tp_rank_id, expert_shard_type, shared_expert_rank_num, moe_expert_num,
                    scales, quant_mode, global_bs, expert_scales, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v, zero_expert_num, copy_expert_num, const_expert_num):
            output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(
                x=x,
                expert_ids=expert_ids,
                group_ep=group_ep,
                group_tp=group_tp,
                ep_world_size=ep_world_size,
                tp_world_size=tp_world_size,
                ep_rank_id=ep_rank_id,
                tp_rank_id=tp_rank_id,
                expert_shard_type=expert_shard_type,
                shared_expert_rank_num=shared_expert_rank_num,
                moe_expert_num=moe_expert_num,
                scales=scales,
                quant_mode=quant_mode,
                global_bs=global_bs,
                elastic_info=elastic_info,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num
            )
            expand_x_npu, _, assist_info_for_combine_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
            if expand_x_npu.dtype == torch.int8:
                expand_x_npu = expand_x_npu.to(input_dtype)

            output_combine_npu = torch_npu.npu_moe_distribute_combine_v2(
                expand_x=expand_x_npu,
                expert_ids=expert_ids,
                assist_info_for_combine=assist_info_for_combine_npu,
                ep_send_counts=ep_recv_counts_npu,
                tp_send_counts=tp_recv_counts_npu,
                expert_scales=expert_scales,
                group_ep=group_ep,
                group_tp=group_tp,
                ep_world_size=ep_world_size,
                tp_world_size=tp_world_size,
                ep_rank_id=ep_rank_id,
                tp_rank_id=tp_rank_id,
                expert_shard_type=expert_shard_type,
                shared_expert_rank_num=shared_expert_rank_num,
                moe_expert_num=moe_expert_num,
                global_bs=global_bs,
                elastic_info=elastic_info,
                ori_x=x,
                const_expert_alpha_1=const_expert_alpha_1,
                const_expert_alpha_2=const_expert_alpha_2,
                const_expert_v=const_expert_v,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num
            )
            x = output_combine_npu
            x_combine_res = output_combine_npu
            return [x_combine_res, output_combine_npu]

    def gen_const_expert_alpha_1():
        const_expert_alpha_1 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1

    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_2

    def gen_const_expert_v():
        const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_v

    def get_new_group(rank):
        for i in range(tp_world_size):
            ep_ranks = [x * tp_world_size + i for x in range(ep_world_size)]
            ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group_t = ep_group
                print(f"rank:{rank} ep_ranks:{ep_ranks}")
        for i in range(ep_world_size):
            tp_ranks = [x + tp_world_size * i for x in range(tp_world_size)]
            tp_group = dist.new_group(backend="hccl", ranks=tp_ranks)
            if rank in tp_ranks:
                tp_group_t = tp_group
                print(f"rank:{rank} tp_ranks:{tp_ranks}")
        return ep_group_t, tp_group_t

    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info

    def warm_up_dispatch(rank, group_ep, group_tp):
        x_warm_up = torch.empty(size=[1, h], dtype=input_dtype).uniform_(-1024, 1024).to(input_dtype).npu()
        expert_ids_warm_up = torch.arange(0, k, dtype=torch.int32).unsqueeze(0).npu()

        dispatch_kwargs_before = get_dispatch_kwargs_warmup(
            x_warm_up=x_warm_up,
            expert_ids_warm_up=expert_ids_warm_up,
            group_ep=group_ep,
            group_tp=group_tp,
            ep_rank_id=rank // tp_world_size,
            tp_rank_id=rank % tp_world_size,
        )

        (
            expand_x, dynamic_scales, expand_idx,
            expert_token_nums, ep_recv_counts, tp_recv_counts, _
        ) = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs_before)
        return expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts

    def get_dispatch_kwargs_warmup(
        x_warm_up, expert_ids_warm_up, group_ep, group_tp, ep_rank_id, tp_rank_id,
    ):
        x_warm_up = x_warm_up.to(input_dtype).npu()
        expert_ids_warm_up = expert_ids_warm_up.to(torch.int32).npu()

        return {
            'x': x_warm_up,
            'expert_ids': expert_ids_warm_up,
            'x_active_mask': None,
            'group_ep': group_ep,
            'group_tp': group_tp,
            'ep_rank_id': ep_rank_id,
            'tp_rank_id': tp_rank_id,
            'ep_world_size': ep_world_size,
            'tp_world_size': tp_world_size,
            'expert_shard_type': 0,
            'shared_expert_num': 0,
            'shared_expert_rank_num': shared_expert_rank_num,
            'moe_expert_num': moe_expert_num,
            'scales': None,
            'quant_mode': 2,
            'global_bs': 16,
        }

    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(
            backend='hccl',
            rank=rank,
            world_size=world_size,
            init_method=f'tcp://{master_ip}:{port}'
        )
        ep_group, tp_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        tp_hcomm_info = get_hcomm_info(rank, tp_group)

        # Create input tensors
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = torch.tensor([
            [0, 8, 4, 1, 6, 12, 14, 17],
            [14, 10, 7, 3, 0, 12, 11, 17],
            [12, 0, 5, 11, 19, 4, 6, 18],
            [17, 3, 4, 10, 18, 0, 1, 2],
            [13, 16, 9, 10, 15, 6, 7, 14],
            [17, 15, 14, 8, 16, 18, 3, 12],
            [4, 12, 2, 17, 15, 3, 9, 10],
            [16, 7, 12, 9, 18, 3, 19, 17]
        ], dtype=torch.int32).npu()

        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + moe_expert_num, h) if shared_expert_rank_num else (moe_expert_num, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None

        elastic_info = None
        const_expert_alpha_1 = gen_const_expert_alpha_1().npu()
        const_expert_alpha_2 = gen_const_expert_alpha_2().npu()
        const_expert_v = gen_const_expert_v().npu()
        out = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)

        model = MOE_DISTRIBUTE_GRAPH_Model()
        model = model.npu()
        npu_backend = torchair.get_npu_backend()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        output = model.forward(
            x, expert_ids, ep_hcomm_info, tp_hcomm_info, ep_world_size, tp_world_size,
            rank // tp_world_size, rank % tp_world_size, 0, shared_expert_rank_num, moe_expert_num, scales,
            quant_mode, globalBS, expert_scales, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
            zero_expert_num, copy_expert_num, const_expert_num
        )
        torch.npu.synchronize()
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')

        time.sleep(10)

    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"global_bs={globalBS}")
        print(f"shared_expert_rank_num={shared_expert_rank_num}")
        print(f"moe_expert_num={moe_expert_num}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
        print(f"tp_world_size={tp_world_size}", flush=True)
        print(f"ep_world_size={ep_world_size}", flush=True)

        if tp_world_size != 1 and local_moe_expert_num > 1:
            print("unSupported tp = 2 and local moe > 1")
            exit(0)

        if shared_expert_rank_num > ep_world_size:
            print("shared_expert_rank_num cannot be greater than ep_world_size")
            exit(0)

        if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
            print("ep_world_size must be an integer multiple of shared_expert_rank_num")
            exit(0)

        if moe_expert_num % moe_rank_num != 0:
            print("moe_expert_num must be an integer multiple of moe_rank_num")
            exit(0)

        p_list = []
        for rank in range(rank_per_dev):
            p = Process(target=run_npu_process, args=(rank,))
            p_list.append(p)

        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

        print("run npu success.")
    ```
