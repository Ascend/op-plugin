# torch_npu.npu_moe_distribute_combine_v2

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

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

## Function<a name="en-us_topic_0000002168254826_section14441124184110"></a>

- **Function**: This API must be used together with [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md). It returns data along the reverse path of the data collected by the `npu_moe_distribute_dispatch_v2` operator.
    - Data aggregation is supported: reduce_scatterv communication is performed first, then alltoallv communication, and finally the received data is aggregated (multiplied by weights and added).
    - Special expert scenarios (zero expert, copy expert, and const expert) are supported. Together with `assist_info_for_combine` of dispatch_v2, more efficient all-rank synchronization is achieved.

- Formulas:
    - Data aggregation:

        $$
        rs\_out = ReduceScatterV(expand\_x)
        $$

        $$
        ata\_out = AlltoAllv(rs\_out)
        $$

        $$
        x = Sum(expert\_scales * ata\_out + expert\_scales * shared\_expert\_x)
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

## Prototype<a name="en-us_topic_0000002168254826_section45077510411"></a>

```python
torch_npu.npu_moe_distribute_combine_v2(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num, *, tp_send_counts=None, x_active_mask=None, expand_scales=None, shared_expert_x=None, elastic_info=None, ori_x=None, const_expert_alpha_1=None, const_expert_alpha_2=None, const_expert_v=None, performance_info=None, group_tp="", tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, global_bs=0, comm_quant_mode=0, comm_alg="", zero_expert_num=0, copy_expert_num=0, const_expert_num=0) -> Tensor
```

## Parameters<a name="en-us_topic_0000002168254826_section112637109429"></a>

- **`expand_x`** (`Tensor`): Required. Token features extended based on `expert_ids`. This parameter must be 2D with shape `(max(tp_world_size, 1) * A, H)`. The data layout can be ND. Non-contiguous tensors are supported. The data type can be `torch.bfloat16` or `torch.float16`.
    <!-- npu="910b" id69 -->
    - Atlas A2 products do not support the shared expert scenario.
    <!-- end id69 -->

- **`expert_ids`** (`Tensor`): Required. Top-K expert indices for each token. This parameter must be 2D with shape `(BS, K)`. The data type can be `torch.int32`. The data layout can be ND. Non-contiguous tensors are supported. This parameter corresponds to the `expert_ids` input of [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md). The tensor values range from [0, moe_expert_num), and the K values in the same row must not be repeated.
- **`assist_info_for_combine`** (`Tensor`): Required. Number of tokens dispatched to each expert. This parameter must be 1D with shape `(A * 128,)`. The data type can be `torch.int32`. The data layout can be ND. Non-contiguous tensors are supported. This parameter corresponds to the `assist_info_for_combine` output of [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md).
- **`ep_send_counts`** (`Tensor`): Required. Token count sent from each expert on the current rank to each rank within the Expert Parallelism (EP) communication domain, where token counts are represented as prefix sums. This parameter must be 1D. The data type can be `torch.int32`. The data layout is ND. Non-contiguous tensors are supported. This parameter corresponds to the `ep_recv_counts` output of [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md).

    <!-- npu="910b" id4 -->
    - <term>Atlas A2 products</term>: The shape must be `(moe_expert_num + 2 * global_bs * K * server_num,)`. The first `moe_expert_num` elements indicate the token counts received by each expert on the current rank from other ranks within the EP communication domain, represented as prefix sums. The remaining `2 * global_bs * K * server_num` elements are used to store the token counts that can be reduced in advance during the combine operation and the communication buffer offsets before executing inter-server and intra-server communication. When the value of `global_bs` is `0`, the value is calculated as `bs * ep_world_size`.
    <!-- end id4 -->
    <!-- npu="950,A3" id5 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The shape must be (ep_world_size*max(tp_world_size, 1)*local_expert_num,).
    <!-- end id5 -->

- **`expert_scales`** (`Tensor`): Required. Weights of the top-K experts for each token.

    <!-- npu="A3,910b" id6 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: This parameter must be 2D with shape `(BS, K)`, where no weight coefficient needs to be multiplied for shared experts and the values are added directly. The data type can be `torch.float`. The data layout must be ND. Non-contiguous tensors are supported.
    <!-- end id6 -->
    <!-- npu="950" id7 -->
    - <term>Ascend 950DT products</term>: An empty tensor or a 2D tensor can be passed. When a non-empty tensor is passed, the shape is `(BS, K)`, where no weight coefficient needs to be multiplied for shared experts and the values are added directly. The data type can be `torch.float`. The data layout must be ND. Non-contiguous tensors are supported.
    <!-- end id7 -->

- **`group_ep`** (`str`): Required. EP communication domain name used for expert parallelism. The string length range is [1, 128).

    <!-- npu="A3" id8 -->
    - <term>Atlas A3 products</term>: The value must differ from `group_tp`.
    <!-- end id8 -->

- **`ep_world_size`** (`int`): Required. Size of the EP communication domain.

    <!-- npu="910b" id9 -->
    - <term>Atlas A2 products</term>: The value range of `ep_world_size` is as follows:
        - When `comm_alg` is set to `"fullmesh"`, the value range of `ep_world_size` is `16`, `32`, `64`, `128`, or `256`.
        - When `comm_alg` is set to `"hierarchy"`: `16`, `32`, or `64`.
    <!-- end id9 -->
    <!-- npu="A3" id10 -->
    - <term>Atlas A3 products</term>: The value range is [2, 768]. When `comm_alg` is set to `"hierarchy"`, the value range is [16, 256] and the value must be a multiple of 16.
    <!-- end id10 -->
    <!-- npu="950" id11 -->
    - <term>Ascend 950DT products</term>: The value range is [2, 1024].
    <!-- end id11 -->

- **`ep_rank_id`** (`int`): Required. Rank ID of the current rank within the EP communication domain. The value range is [0, `ep_world_size`). The `ep_rank_id` values of all ranks within the identical EP communication domain must be unique.
- **`moe_expert_num`** (`int`): Required. Number of MoE experts. The value range is [1, 1024], and the condition `moe_expert_num % (ep_world_size - shared_expert_rank_num) == 0` must be satisfied.

    <!-- npu="A3" id12 -->
    - <term>Atlas A3 products</term>: When `comm_alg` is set to `"hierarchy"`, the value range is (0, 512].
    <!-- end id12 -->

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`tp_send_counts`** (`Tensor`): Optional. Amount of data sent from each expert on the current rank to each rank within the Tensor Parallelism (TP) communication domain. This parameter corresponds to the `tp_recv_counts` output of [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md).

    <!-- npu="950,910b" id13 -->
    - <term>Atlas A2 products</term>, <term>Ascend 950DT products</term>: The TP communication domain is not supported. Use the default input.
    <!-- end id13 -->
    <!-- npu="A3" id14 -->
    - <term>Atlas A3 products</term>: The TP communication domain is supported. This parameter must be 1D with shape `(tp_world_size,)`. The data type can be `torch.int32`. The data layout must be ND. Non-contiguous tensors are supported.
    <!-- end id14 -->

- **`x_active_mask`** (`Tensor`): Optional. Controls whether tokens participate in communication.

    <!-- npu="910b" id15 -->
    - <term>Atlas A2 products</term>:
        - When `comm_alg` is set to "fullmesh", this parameter must be 1D with shape `(BS,)`. The data type can be `torch.bool`. The data layout must be ND. Non-contiguous tensors are supported. The value `true` indicates that the corresponding token participates in communication, and all `true` values must precede any `false` values. For example, `{true, false, true}` is an invalid input. By default, all tokens participate in communication. When the `BS` values differ across ranks, all tokens must be valid.
        - When `comm_alg` is set to "hierarchy", it is not supported in this version. Use the default value.
    <!-- end id15 -->
    <!-- npu="950,A3" id16 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: This parameter must be 1D or 2D. When the input is 1D, the shape is `(BS,)`; when the input is 2D, the shape is `(BS, K)`. The data type can be `torch.bool`. The data layout is $ND$. Non-contiguous tensors are supported. When the input is 1D, the value `true` indicates that the corresponding token participates in communication, and all `true` values must precede any `false` values. For example, `{true, false, true}` is an invalid input. When the input is 2D, the value `true` indicates that the `expert_ids` entry corresponding to the current token participates in communication. If all K `torch.bool` values for a token are `false`, the token does not participate in communication. By default, all tokens participate in communication. When the `BS` values differ across ranks, all tokens must be valid.
    <!-- end id16 -->

- **`expand_scales`** (`Tensor`): Optional. This parameter corresponds to the `expand_scales` output of [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md).

    <!-- npu="910b" id17 -->
    - <term>Atlas A2 products</term>: Required. This parameter must be 1D with shape `(A,)`. The data type can be `torch.float`. The data layout is ND. Non-contiguous tensors are supported.
    <!-- end id17 -->
    <!-- npu="950,A3" id18 -->
    - <term>Atlas A3 products</term> and <term>Ascend 950DT products</term>: This parameter is not supported currently. Use the default value.
    <!-- end id18 -->

- **`shared_expert_x`** (`Tensor`): Optional. The data type must be identical to that of `expand_x`. This parameter is the shared-expert token data that must be added after `combine_v2`. Use it only when the number of shared-expert devices `shared_expert_rank_num` is not `0`.

    <!-- npu="910b" id19 -->
    - <term>Atlas A2 products</term>: Currently, this parameter is not supported. Retain the default value.
    <!-- end id19 -->
    <!-- npu="950,A3" id20 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: This parameter must be a 2D or 3D tensor. When the tensor is 2D, the shape is `(BS, H)`; when the tensor is 3D, the product of the first two dimensions must equal BS, and the third dimension must equal H.
    <!-- end id20 -->

- **`elastic_info`** (`Tensor`): Optional. Reserved parameter, currently not used. Retain the default value `None`.
- **`ori_x`** (`Tensor`): Optional. Token data before FFN processing. This parameter is required when `copy_expert_num` or `const_expert_num` is not `0`.

    <!-- npu="910b" id21 -->
    - <term>Atlas A2 products</term>: Reserved parameter. Not supported in the current version; pass `None`.
    <!-- end id21 -->
    <!-- npu="950,A3" id22 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: You can choose to pass a valid tensor or `None`. When `copy_expert_num` is not 0 or `const_expert_num` is not 0, a valid input must be passed. When valid data is passed, it must be a 2D tensor with the shape `(BS, H)`, and its data type must be consistent with that of `expand_x`. The data layout is $ND$. Non-contiguous tensors are supported.
    <!-- end id22 -->

- **`const_expert_alpha_1`** (`Tensor`): Optional. Calculation coefficient required when `const_expert_num` is not `0`.

    <!-- npu="910b" id23 -->
    - <term>Atlas A2 products</term>: Reserved parameter, currently not supported. Set it to `None`.
    <!-- end id23 -->
    <!-- npu="950,A3" id24 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: You can choose to pass a valid tensor or `None`. When `const_expert_num` is not 0, a valid input must be passed. When valid data is passed, it must be a 2D tensor with the shape `(const_expert_num, H)`, and its data type must be consistent with that of `expand_x`. The data layout is $ND$. Non-contiguous tensors are supported.
    <!-- end id24 -->

- **`const_expert_alpha_2`** (`Tensor`): Optional. Calculation coefficient required when `const_expert_num` is not `0`.

    <!-- npu="910b" id25 -->
    - <term>Atlas A2 products</term>: Reserved parameter, currently not supported. Set it to `None`.
    <!-- end id25 -->
    <!-- npu="950,A3" id26 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: You can choose to pass a valid tensor or `None`. When `const_expert_num` is not 0, a valid input must be passed. When valid data is passed, it must be a 2D tensor with the shape `(const_expert_num, H)`, and its data type must be consistent with that of `expand_x`. The data layout is $ND$. Non-contiguous tensors are supported.
    <!-- end id26 -->

- **`const_expert_v`** (`Tensor`): Optional. Calculation coefficient required when `const_expert_num` is not `0`.

    <!-- npu="910b" id27 -->
    - <term>Atlas A2 products</term>: Reserved parameter, currently not supported. Set it to `None`.
    <!-- end id27 -->
    <!-- npu="950,A3" id28 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: You can choose to pass a valid tensor or `None`. When `const_expert_num` is not 0, a valid input must be passed. When valid data is passed, it must be a 2D tensor with the shape `(const_expert_num, H)`, and its data type must be consistent with that of `expand_x`. The data layout is $ND$. Non-contiguous tensors are supported.
    <!-- end id28 -->

- **`performance_info`** (`Tensor`): Optional. Communication wait time of the current rank for data from other ranks, in microseconds (us). The communication time incurred by each operator call is accumulated into this tensor. The operator does not clear the tensor automatically. Therefore, before enabling this tensor to record communication time, you must clear it manually. Passing `None` disables communication-time recording. When a valid tensor is provided, this parameter must be 1D with shape `(ep_world_size,)`. The data type can be `torch.int64`. The data layout must be ND. Non-contiguous tensors are supported.
- **`group_tp`** (`str`): Optional. TP communication domain name, that is, the communication domain for data parallelism. This parameter is required only when TP domain communication exists. If there is no TP domain communication, use the default value "".

    <!-- npu="910b" id29 -->
    - <term>Atlas A2 products</term>: In eager mode, use the default value. In graph mode, the value must be identical to that of `group_ep`.
    <!-- end id29 -->
    <!-- npu="A3" id30 -->
    - <term>Atlas A3 products</term>: The string length value range is [0, 128). The value of this parameter must differ from `group_ep`. An empty value is supported only when there is no TP domain.
    <!-- end id30 -->
    <!-- npu="950" id31 -->
    - <term>Ascend 950DT products</term>: TP domain communication is not supported. Use the default input.
    <!-- end id31 -->

- **`tp_world_size`** (`int`): Optional. Size of the TP communication domain. This parameter is required only when TP domain communication is involved.

    <!-- npu="910b" id32 -->
    - <term>Atlas A2 products</term>: TP domain communication is not supported. Use the default value `0`.
    <!-- end id32 -->
    <!-- npu="A3" id33 -->
    - <term>Atlas A3 products</term>: When TP domain communication is involved, the value range is [0, 2]. The values `0` and `1` indicate that there is no TP domain communication, and the value `2` indicates that there is TP domain communication.
    <!-- end id33 -->
    <!-- npu="950" id34 -->
    - <term>Ascend 950DT products</term>: TP domain communication is not supported. Use the default value `0`.
    <!-- end id34 -->

- **`tp_rank_id`** (`int`): Optional. Rank ID of the current rank within the TP communication domain. This parameter is required only when TP domain communication is involved.

    <!-- npu="910b" id35 -->
    - <term>Atlas A2 products</term>: TP domain communication is not supported. Use the default value `0`.
    <!-- end id35 -->
    <!-- npu="A3" id36 -->
    - <term>Atlas A3 products</term>: When TP domain communication is involved, the value range is [0, 1]. The `tp_rank_id` of each rank in the same TP communication domain must be unique. If there is no TP domain communication, pass `0`.
    <!-- end id36 -->
    <!-- npu="950" id37 -->
    - <term>Ascend 950DT products</term>: TP domain communication is not supported. Use the default value `0`.
    <!-- end id37 -->

- **`expert_shard_type`** (`int`): Optional. Layout type of shared-expert ranks.

    <!-- npu="910b" id38 -->
    - <term>Atlas A2 products</term>: Currently, this parameter is not supported. Retain the default value.
    <!-- end id38 -->
    <!-- npu="950,A3" id39 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: Currently only 0 is supported, indicating that shared-expert ranks are placed before MoE expert ranks.
    <!-- end id39 -->

- **`shared_expert_num`** (`int`): Optional. Number of shared experts, where a shared expert can be replicated and deployed across multiple ranks.

    <!-- npu="910b" id40 -->
    - <term>Atlas A2 products</term>: This parameter is not supported currently. Use the default value 1.
    <!-- end id40 -->
    <!-- npu="950,A3" id41 -->
    - <term>Atlas A3 products</term> and <term>Ascend 950DT products</term>: When the batch sizes differ across ranks, passing `max_bs * ep_world_size` is supported, where `max_bs` indicates the maximum batch size of a single rank. When the batch sizes are identical across ranks, passing `0` or `BS * ep_world_size` is supported.
    <!-- end id41 -->

- **`shared_expert_rank_num`** (`int`): Optional. Number of shared-expert ranks.

    <!-- npu="910b" id42 -->
    - <term>Atlas A2 products</term>: Expert sharing is not supported. Retain the default value.
    <!-- end id42 -->
    <!-- npu="950,A3" id43 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The value range is [0, ep_world_size). A value of 0 indicates that there is no shared expert; a non-zero value requires `shared_expert_rank_num % shared_expert_num = 0`.
    <!-- end id43 -->

- **`global_bs`** (`int`): Optional. Global batch size within the EP communication domain. When the batch sizes differ across ranks, passing `max_bs * ep_world_size` is supported, where `max_bs` indicates the maximum batch size of a single rank. When the batch sizes are identical across ranks, passing `0` or `BS * ep_world_size` is supported.
- **`comm_quant_mode`** (`int`): Optional. Communication quantization type.

    <!-- npu="910b" id44 -->
    - <term>Atlas A2 products</term>: The value can be `0` or `2`. `0` disables quantization during communication, and `2` enables `torch.int8` quantization during communication. The value `2` is supported only when `comm_alg` is set to `"hierarchy"`, or when `HCCL_INTRA_PCIE_ENABLE=1`, `HCCL_INTRA_ROCE_ENABLE=0`, and the driver version is not earlier than 25.0.RC1.1.
    <!-- end id44 -->
    <!-- npu="A3" id45 -->
    - <term>Atlas A3 products</term>: The value can be `0` or `2`. `0` disables quantization during communication, and `2` enables `torch.int8` quantization during communication. `torch.int8` quantization can be enabled only when `tp_world_size` is not `2`.
    <!-- end id45 -->
    <!-- npu="950" id46 -->
    - <term>Ascend 950DT products</term>: The value range is 0, 2, 3, or 4. 0 indicates no quantization; 2 indicates `torch.int8` quantization; 3 indicates mxfp quantization with output type `torch.float8_e5m2`; 4 indicates mxfp quantization with output type `torch.float8_e4m3fn`.
    <!-- end id46 -->

- **`comm_alg`** (`str`): Optional. Communication-optimized memory-layout algorithm.

    <!-- npu="910b" id47 -->
    - <term>Atlas A2 products</term>: The current version supports `""`, `"fullmesh"`, and `"hierarchy"`. You are advised to use `"hierarchy"` together with driver version 25.0.RC1.1 or later.
        - `""`: If `HCCL_INTRA_PCIE_ENABLE=1` and `HCCL_INTRA_ROCE_ENABLE=0`, the `"hierarchy"` algorithm is used. Otherwise, the `"fullmesh"` algorithm is used. This mode is not recommended.
        - `"fullmesh"`: Token data is sent directly back to the target rank through RDMA.
        - `"hierarchy"`: Token data is transmitted in two stages: intra-server and inter-server. The same token data is first aggregated and summed within a server, and then transmitted across servers to reduce the data volume of inter-server communication.
    <!-- end id47 -->
    <!-- npu="950,A3" id48 -->
    - <term>Atlas A3 products</term> and <term>Ascend 950DT products</term>: TP domain communication is not supported. Use the default value `0`.
    <!-- end id48 -->

- **`zero_expert_num`** (`int`): Optional. Number of zero experts.

    <!-- npu="910b" id49 -->
    - <term>Atlas A2 products</term>: Not supported in the current version; pass `0`.
    <!-- end id49 -->
    <!-- npu="950,A3" id50 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The value range is [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid zero-expert IDs are in the range [moe_expert_num, moe_expert_num+zero_expert_num).
    <!-- end id50 -->

- **`copy_expert_num`** (`int`): Optional. Number of copy experts.

    <!-- npu="910b" id51 -->
    - <term>Atlas A2 products</term>: Not supported in the current version; pass `0`.
    <!-- end id51 -->
    <!-- npu="950,A3" id52 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The value range is [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid copy-expert IDs are in the range [moe_expert_num, moe_expert_num+zero_expert_num+copy_expert_num).
    <!-- end id52 -->

- **`const_expert_num`** (`int`): Optional. Number of constant experts.

    <!-- npu="910b" id53 -->
    - <term>Atlas A2 products</term>: This parameter is currently not supported. Set it to `0`.
    <!-- end id53 -->
    <!-- npu="950,A3" id54 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The value range is [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. Valid constant-expert IDs are in the range [moe_expert_num, moe_expert_num+zero_expert_num+copy_expert_num+const_expert_num).
    <!-- end id54 -->

## Return Values<a name="en-us_topic_0000002168254826_section22231435517"></a>

`Tensor`

Processed tokens. This parameter must be 2D with shape `(BS, H)`. The data type can be `torch.bfloat16` or `torch.float16`. The data type must be identical to that of the input `expand_x`. The data layout can be ND. Non-contiguous tensors are not supported.

## Constraints<a name="en-us_topic_0000002168254826_section12345537164214"></a>

- This API can be used in inference scenarios.
- Communication mode constraints:

    <!-- npu="950" id55 -->
    - <term>Ascend 950DT products</term>: Only UB Memory communication is supported.
    <!-- end id55 -->

- This API supports graph mode. `npu_moe_distribute_dispatch_v2` and `npu_moe_distribute_combine_v2` must be used together.
- Element values inside the `assist_info_for_combine`, `ep_recv_counts`, `tp_recv_counts`, and `expand_scales` tensors output by `npu_moe_distribute_dispatch_v2` can vary across different product models, communication algorithms, or software versions. When using this API, pass these tensors directly to the corresponding parameters of `npu_moe_distribute_combine_v2`. Other service logic of the model must not depend on them.
- Values of the `group_ep`, `ep_world_size`, `moe_expert_num`, `group_tp`, `tp_world_size`, `expert_shard_type`, `shared_expert_num`, `shared_expert_rank_num`, and `global_bs` parameters must remain identical across all ranks during the API execution process. In addition, the values of `group_ep`, `ep_world_size`, `group_tp`, `tp_world_size`, `expert_shard_type`, and `global_bs` must also remain identical across different layers in the network and must match the corresponding parameters in [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md).

<!-- npu="A3" id56 -->
- <term>Atlas A3 products</term>: In this scenario, a single rank contains dual dies. Therefore, "this rank" in the parameter description indicates a single die.
<!-- end id56 -->
- moe\_expert\_num + zero\_expert\_num + copy\_expert\_num + const\_expert\_num < MAX\_INT32.
- Variables used in parameter tensor shapes:
    - `A`: Maximum number of tokens that can be received by the current rank. The value range is as follows:
        - For shared experts: `A = BS * ep_world_size * shared_expert_num/shared_expert_rank_num`.
        - For MoE experts, when `global_bs` is `0`: `A >= BS * ep_world_size * min(local_expert_num, K)`. When `global_bs` is not `0`: `A >= global_bs * min(local_expert_num, K)`.

    - `H`: Hidden layer size.

        <!-- npu="910b" id57 -->
        - <term>Atlas A2 products</term>: The value range of `H` is as follows:
            - When `comm_alg` is set to `"fullmesh"`, the value range of `H` is (0, 7168], and `H` must be a multiple of 32.
            - When `comm_alg` is set to `"hierarchy"` and the driver version is not earlier than 25.0.RC1.1, the value range of `H` is (0, 10 * 1024], and `H` must be a multiple of 32.
        <!-- end id57 -->
        <!-- npu="950,A3" id58 -->
        - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The value range is [1024, 8192].
        <!-- end id58 -->

    - `BS`: Number of tokens to be sent.

        <!-- npu="910b" id59 -->
        - <term>Atlas A2 products</term>: The value range is 0 < BS <= 256.
        <!-- end id59 -->
        <!-- npu="950,A3" id60 -->
        - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The value range is 0 < BS <= 512.
        <!-- end id60 -->

    - `K`: Number of top-K experts selected. The value range is 0 < `K` <= 16, and the condition 0 < `K` <= `moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num` must be satisfied.
    - `server_num`: Number of server nodes. Valid values are `2`, `4`, or `8`.

        <!-- npu="910b" id61 -->
        - <term>Atlas A2 products</term>: This variable is used only in shapes for this scenario.
        <!-- end id61 -->

    - `local_expert_num`: Number of experts on the current rank.
        - For shared-expert ranks, `local_expert_num = 1`.
        - For MoE expert ranks, `local_expert_num = moe_expert_num/(ep_world_size - shared_expert_rank_num)`. When `local_expert_num > 1`, communication in the TP domain is not supported.

        <!-- npu="950,A3" id62 -->
        - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The condition `0 < local_expert_num * ep_world_size <= 2048` must be satisfied.
        <!-- end id62 -->

- HCCL communication domain buffer size:

    Before calling this API, verify that the configured HCCL communication domain buffer size is reasonable. The unit is MB, and the default value is `200` MB if not configured.

    <!-- npu="910b" id63 -->
    - <term>Atlas A2 products</term>: This scenario supports configuration through the `HCCL_BUFFSIZE` environment variable.
        - When `comm_alg` is set to `""`: `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE` take effect only under this configuration. The `"fullmesh"` or `"hierarchy"` formula is selected based on the `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE` configurations.
        - When `comm_alg` is set to `"fullmesh"`: The configured size must be greater than or equal to `2 * (BS * ep_world_size * min(local_expert_num, K) * H * sizeof(`torch.uint16`) + 2MB)`.
        - When `comm_alg` is set to `"hierarchy"`: The configured size must be equal to `moe_expert_num * BS * (H * sizeof(dtype_x) + 4 * ((K + 7)/8 * 8) * sizeof(`torch.uint32`)) + 4MB + 100MB`, and the condition `moe_expert_num/(ep_world_size - shared_expert_rank_num) <= 24` is not required.
    <!-- end id63 -->
    <!-- npu="A3" id64 -->
    - <term>Atlas A3 products</term>: This scenario supports configuration not only through the environment variable `HCCL_BUFFSIZE`, but also through [`hccl_buffer_size`](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/docs/en/performance_tuning/performance_tuning_methods/communication_basics_overview.md#hccl_buffer_size).
        - Within the EP communication domain: The configured size must be greater than or equal to 2, and it must satisfy the condition `>= 2 * (local_expert_num * max_bs * ep_world_size * Align512(Align32(2 * h) + 64) + (K + shared_expert_num) * max_bs * Align512(2 * h))`.
        - Within the TP communication domain: The value must be greater than or equal to `(A * Align512(Align32(h * 2) + 44) + A * Align512(h * 2)) * 2`.
        - The alignment functions are defined as follows: $Align480(x) = ((x + 480 - 1)/480) * 512$, $Align512(x) = ((x + 512 - 1)/512) * 512$, $Align32(x) = ((x + 32 - 1)/32) * 32$.
    <!-- end id64 -->
    <!-- npu="950" id65 -->
    - <term>Ascend 950DT products</term>: The value must be >=2 and must satisfy >= 2\*(local_expert_num\*max_bs\*ep_world_size\*Align512(Align32(2\*H)+64)+(K+shared_expert_num)\*max_bs\*Align512(2\*H)) + 1024 \* 1024 \* 10, where 480Align512(x) = ((x+480-1)/480)\*512, Align512(x) = ((x+512-1)/512)\*512, and Align32(x) = ((x+32-1)/32)\*32.
    <!-- end id65 -->

<!-- npu="910b" id66 -->
- <term>Atlas A2 products</term>: These environment variables are no longer recommended. You are advised to set `comm_alg` to `"hierarchy"`.
<!-- end id66 -->
- In formulas in this document, the division sign (/) indicates integer division.
    - The `npu_moe_distribute_dispatch_v2` and `npu_moe_distribute_combine_v2` operators within a single model must operate in the same EP communication domain, which must not include other operators.
    - The `npu_moe_distribute_dispatch_v2` and `npu_moe_distribute_combine_v2` operators within a single model must either operate in the same TP communication domain or both operate without a TP communication domain. When a TP communication domain is involved, this domain must not include other operators.

    <!-- npu="A3" id67 -->
    - <term>Atlas A3 products</term>: Nodes in a communication domain must reside within the same SuperPoD. Cross-SuperPoD deployment is not supported.
    <!-- end id67 -->

<!-- npu="910b" id68 -->
- Networking constraints:

    - <term>Atlas A2 products</term>: In multi-server deployments, only switch-based networking is supported. Direct connections between two servers are not supported.
<!-- end id68 -->

- In the formula, "/" indicates integer division.

## Examples<a name="en-us_topic_0000002168254826_section14459801435"></a>

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
