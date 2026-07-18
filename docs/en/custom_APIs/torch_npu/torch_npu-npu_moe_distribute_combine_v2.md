# torch\_npu.npu\_moe\_distribute\_combine\_v2<a name="en-us_topic_0000002309174912"></a>

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>          |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |

## Function<a name="en-us_topic_0000002168254826_section14441124184110"></a>

- Description:

    This API must be used together with [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md), serving as the reverse data collection path of the `npu_moe_distribute_dispatch_v2` operator.
     - This API supports data aggregation by fusing `reduce_scatterv` communication, `alltoallv` communication, and final data aggregation (multiplying the corresponding weights and summing the results).
     - It also supports special expert scenarios.
- Formulas:
    - Data aggregation:

      $rs\_out = ReduceScatterV(expand\_x)$

      $ata\_out = AlltoAllv(rs\_out)$

      $x = Sum(expert\_scales * ata\_out + expert\_scales * shared\_expert\_x)$

    - Special expert scenarios:

      Zero expert scenario (`zero_expert_num ≠ 0`):

      $$Moe(ori\_x)=0$$

      Copy expert scenario (`copy_expert_num ≠ 0`):

      $$Moe(ori\_x)=ori\_x$$

      Constant expert scenario (`const_expert_num ≠ 0`):

      $Moe(ori\_x)=const\_expert\_alpha\_1*ori\_x+const\_expert\_alpha\_2*const\_expert\_v$

## Prototype<a name="en-us_topic_0000002168254826_section45077510411"></a>

```python
torch_npu.npu_moe_distribute_combine_v2(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num, *, tp_send_counts=None, x_active_mask=None, expand_scales=None, shared_expert_x=None, elastic_info=None, ori_x=None, const_expert_alpha_1=None, const_expert_alpha_2=None, const_expert_v=None, performance_info=None, group_tp="", tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, global_bs=0, comm_quant_mode=0, comm_alg="", zero_expert_num=0, copy_expert_num=0, const_expert_num=0) -> Tensor
```

## Parameters<a name="en-us_topic_0000002168254826_section112637109429"></a>

- **`expand_x`** (`Tensor`): Required. Token features expanded according to `expert_ids`. This parameter must be 2D with shape `(max(tp_world_size, 1) * A, H)`. The data type can be `bfloat16`. The data layout can be ND. Non-contiguous tensors are supported.
    - Atlas A2 training products/Atlas A2 inference products: Shared-expert scenarios is not supported.

- **`expert_ids`** (`Tensor`): Required. Top-K expert indices for each token. This parameter must be 2D with shape `(BS, K)`. The data type can be `int32`. The data layout can be ND. Non-contiguous tensors are supported. This parameter corresponds to the `expert_ids` input of [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md). The value range of elements inside the tensor is `[0, moe_expert_num)`, and the top-K values within the identical row must be unique.
- **`assist_info_for_combine`** (`Tensor`): Required. Number of tokens dispatched to each expert. This parameter must be 1D with shape `(A * 128,)`. The data type can be `int32`. The data layout can be ND. Non-contiguous tensors are supported. This parameter corresponds to the `assist_info_for_combine` output of [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md).

- **`ep_send_counts`** (`Tensor`): Required. Number of tokens sent from each expert on the current rank to each rank within the Expert Parallelism (EP) communication domain, where token counts are represented as prefix sums. This parameter must be a 1D tensor. The data type can be `int32`. The data layout can be ND. Non-contiguous tensors are supported. This parameter corresponds to the `ep_recv_counts` output of [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md).
    - Atlas A2 training products/Atlas A2 inference products: The shape must be `(moe_expert_num + 2 * global_bs * K * server_num,)`. The first `moe_expert_num` elements indicate the token counts received by each expert on the current rank from other ranks within the EP communication domain, represented as prefix sums. The remaining `2 * global_bs * K * server_num` elements are used to store the token counts that can be reduced in advance during the combine operation and the communication buffer offsets before executing inter-server and intra-server communication. When the value of `global_bs` is `0`, the value is calculated as `bs * ep_world_size`.
    - Atlas A3 training products/Atlas A3 inference products: The shape must be `(ep_world_size * max(tp_world_size, 1) * local_expert_num,)`.

- **`expert_scales`** (`Tensor`): Required. Weights of the top-K experts for each token. This parameter must be 2D with shape `(BS, K)`. Shared-expert configurations do not require a weight factor and are summed directly. The data type can be `float32`. The data layout can be ND. Non-contiguous tensors are supported.
- **`group_ep`** (`str`): Required. EP communication domain name used for expert parallelism. The string length range is [1, 128). On Atlas A3 training products/Atlas A3 inference products, the value of this parameter must differ from `group_tp`.
- **`ep_world_size`** (`int`): Required. Size of the EP communication domain.
    - Atlas A2 training products/Atlas A2 inference products: The valid values are as follows:
         - When `comm_alg` is set to `"fullmesh"`: `2`, `3`, `4`, `5`, `6`, `7`, `8`, `16`, `32`, `64`, `128`, or `256`.
         - When `comm_alg` is set to `"hierarchy"`: `16`, `32`, or `64`.
    - Atlas A3 training products/Atlas A3 inference products: The value range is [2, 768]. When `comm_alg` is set to `"hierarchy"`, the value range is [16, 256] and the value must be a multiple of 16.

- **`ep_rank_id`** (`int`): Required. Rank ID of the current rank within the EP communication domain. The value range is [0, `ep_world_size`). The `ep_rank_id` values of all ranks within the identical EP communication domain must be unique.
- **`moe_expert_num`** (`int`): Required. Number of MoE experts. The value range is [1, 1024], and the condition `moe_expert_num % (ep_world_size - shared_expert_rank_num) == 0` must be satisfied.
    - Atlas A3 training products/Atlas A3 inference products: When `comm_alg` is set to `"hierarchy"`, the value range is (0, 512].
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`tp_send_counts`** (`Tensor`): Optional. Amount of data sent from each expert on the current rank to each rank within the Tensor Parallelism (TP) communication domain. This parameter corresponds to the `tp_recv_counts` output of [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md).
    - Atlas A2 training products/Atlas A2 inference products: The TP communication domain is not supported. Use the default value.
    - Atlas A3 training products/Atlas A3 inference products: The TP communication domain is supported. This parameter must be 1D with shape `(tp_world_size,)`. The data type can be `int32`. The data layout must be ND. Non-contiguous tensors are supported.

- **`x_active_mask`** (`Tensor`): Optional. Controls whether tokens participate in communication.
    - Atlas A2 training products/Atlas A2 inference products:
         - When `comm_alg` is set to `"fullmesh"`, this parameter must be a 1D or 2D tensor. When the input is 1D, the shape of this parameter is `(BS,)`. When the input is 2D, the shape of this parameter is `(BS, K)`. The data type can be `bool`. The data layout must be ND. Non-contiguous tensors are supported. When the input is 1D, the value `True` indicates that the corresponding token participates in communication, and all `True` values must precede any `False` values. For example, `{True, False, True}` is an invalid input. When the input is 2D, the value `True` indicates that the `expert_ids` entry corresponding to the current token participates in communication. If all $K$ Boolean values for a token are `False`, the token does not participate in communication. By default, all tokens participate in communication. When the `BS` values differ across ranks, all tokens must be valid. Support for 2D tensors is part of the zero-computation expert feature, which is currently experimental and must be used with caution.
         - When `comm_alg` is set to `"hierarchy"`, this parameter is currently not supported. Use the default value `None`.
    - Atlas A3 training products/Atlas A3 inference products: This parameter must be a 1D or 2D tensor. When the input is 1D, the shape of this parameter is `(BS,)`. When the input is 2D, the shape of this parameter is `(BS, K)`. The data type can be `bool`. The data layout must be ND. Non-contiguous tensors are supported. When the input is 1D, the value `True` indicates that the corresponding token participates in communication, and all `True` values must precede any `False` values. For example, `{True, False, True}` is an invalid input. When the input is 2D, the value `True` indicates that the `expert_ids` entry corresponding to the current token participates in communication. If all $K$ Boolean values for a token are `False`, the token does not participate in communication. By default, all tokens participate in communication. When the `BS` values differ across ranks, all tokens must be valid.

- `expand_scales` (`Tensor`): Optional. This parameter corresponds to the `expand_scales` output of [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md).
    - Atlas A2 training products/Atlas A2 inference products: Required. This parameter must be 1D with shape `(A,)`. The data type can be `float32`. The data layout can be ND. Non-contiguous tensors are supported.
    - Atlas A3 training products/Atlas A3 inference products: When `comm_alg` is set to `"hierarchy"`, this parameter must be 1D with shape `(A,)`. The data type can be `float32`. The data layout can be ND. Non-contiguous tensors are supported. When `comm_alg` is set to `""`, this parameter is not supported. Use the default value.

- **`shared_expert_x`** (`Tensor`): Optional. The data type must be identical to that of `expand_x`. This parameter is the shared-expert token data that must be added after `combine_v2`. Use it only when the number of shared-expert devices `shared_expert_rank_num` is `0`.
    - Atlas A2 training products/Atlas A2 inference products: Currently, this parameter is not supported. Retain the default value.
    - Atlas A3 training products/Atlas A3 inference products: This parameter must be a 2D or 3D tensor. When the tensor is 2D, the shape is `(BS, H)`. When the tensor is 3D, the product of the first two dimensions must be equal to `BS`, and the third dimension must be equal to `H`.

- **`elastic_info`** (`Tensor`): Optional. Reserved parameter, currently not used. Retain the default value `None`.

- **`ori_x`** (`Tensor`): Optional. Token data before FFN processing. This parameter is required when `copy_expert_num` or `const_expert_num` is not `0`.
    - Atlas A2 training products/Atlas A2 inference products:
         - When `comm_alg` is set to `"fullmesh"`, you can choose to pass a valid tensor or `None`. When `copy_expert_num` is not 0, a valid tensor must be provided. When a valid tensor is provided, this parameter must be 2D with shape `(BS, H)`. The data type must be identical to that of `expand_x`. The data layout must be ND. Non-contiguous tensors are supported. A valid tensor enables the zero-computation expert feature, which is currently experimental and must be used with caution.
         - When `comm_alg` is set to `"hierarchy"`, this parameter is currently not supported. Use the default value `None`.
    - Atlas A3 training products/Atlas A3 inference products: You can choose to pass a valid tensor or `None`. When `copy_expert_num` or `const_expert_num` is not `0`, a valid tensor must be provided. When a valid tensor is provided, it must be 2D with shape `(BS, H)`. The data type must be identical to that of `expand_x`. The data layout must be ND. Non-contiguous tensors are supported.

- **`const_expert_alpha_1`** (`Tensor`): Optional. Calculation coefficient required when `const_expert_num` is not `0`.
    - Atlas A2 training products/Atlas A2 inference products: Reserved parameter, currently not supported. Set it to `None`.
    - Atlas A3 training products/Atlas A3 inference products: You can choose to pass a valid tensor or `None`. When `const_expert_num` is not `0`, a valid tensor must be provided. When a valid tensor is provided, this parameter must be 2D with shape `(const_expert_num, H)`. The data type must be identical to that of `expand_x`. The data layout must be ND. Non-contiguous tensors are supported.

- **`const_expert_alpha_2`** (`Tensor`): Optional. Calculation coefficient required when `const_expert_num` is not `0`.
    - Atlas A2 training products/Atlas A2 inference products: Reserved parameter, currently not supported. Set it to `None`.
    - Atlas A3 training products/Atlas A3 inference products: You can choose to pass a valid tensor or `None`. When `const_expert_num` is not `0`, a valid tensor must be provided. When a valid tensor is provided, this parameter must be 2D with shape `(const_expert_num, H)`. The data type must be identical to that of `expand_x`. The data layout must be ND. Non-contiguous tensors are supported.

- **`const_expert_v`** (`Tensor`): Optional. Calculation coefficient required when `const_expert_num` is not `0`.
    - Atlas A2 training products/Atlas A2 inference products: Reserved parameter, currently not supported. Set it to `None`.
    - Atlas A3 training products/Atlas A3 inference products: You can choose to pass a valid tensor or `None`. When `const_expert_num` is not `0`, a valid tensor must be provided. When a valid tensor is provided, this parameter must be 2D with shape `(const_expert_num, H)`. The data type must be identical to that of `expand_x`. The data layout must be ND. Non-contiguous tensors are supported.

- **`performance_info`** (`Tensor`): Optional. Communication wait time of the current rank for data from other ranks, in microseconds (us). The communication time incurred by each operator call is accumulated into this tensor. The operator does not clear the tensor automatically. Therefore, before enabling this tensor to record communication time, you must clear it manually. Passing `None` disables communication-time recording. When a valid tensor is provided, this parameter must be 1D with shape `(ep_world_size,)`. The data type can be `int64`. The data layout must be ND. Non-contiguous tensors are supported.

- **`group_tp`** (`string`): Optional. Communication domain name used for tensor parallelism. This parameter is required only when TP domain communication is involved. Otherwise, use the default value `""`.
    - Atlas A2 training products/Atlas A2 inference products: In eager mode, use the default value. In graph mode, the value must be identical to that of `group_ep`.
    - Atlas A3 training products/Atlas A3 inference products: The string length value range is [0, 128). The value of this parameter must differ from `group_ep`. An empty value is supported only when there is no TP domain.

- **`tp_world_size`** (`int`): Optional. Size of the TP communication domain. This parameter is required only when TP domain communication is involved.
    - Atlas A2 training products/Atlas A2 inference products: TP domain communication is not supported. Use the default value `0`.
    - Atlas A3 training products/Atlas A3 inference products: When TP domain communication is involved, the value range is [0, 2]. The values `0` and `1` indicate that there is no TP domain communication, and the value `2` indicates that there is TP domain communication.

- **`tp_rank_id`** (`int`): Optional. Rank ID of the current rank within the TP communication domain. This parameter is required only when TP domain communication is involved.
    - Atlas A2 training products/Atlas A2 inference products: TP domain communication is not supported. Use the default value `0`.
    - Atlas A3 training products/Atlas A3 inference products: When TP domain communication is involved, the value range is [0, 1]. The `tp_rank_id` of each rank in the same TP communication domain must be unique. If there is no TP domain communication, pass `0`.

- **`expert_shard_type`** (`int`): Optional. Layout type of shared-expert ranks.
    - Atlas A2 training products/Atlas A2 inference products: Currently, this parameter is not supported. Retain the default value.
    - Atlas A3 training products/Atlas A3 inference products: Currently, only `0` is supported, indicating that shared-expert ranks are arranged in front of MoE expert ranks.

- **`shared_expert_num`** (`int`): Optional. Number of shared experts, where a shared expert can be replicated and deployed across multiple ranks.
    - Atlas A2 training products/Atlas A2 inference products: Currently, this parameter is not supported. Retain the default value.
    - Atlas A3 training products/Atlas A3 inference products: The value range is [0, 4]. Passing `0` indicates that no shared experts are used, and the default value is `1`.

- **`shared_expert_rank_num`** (`int`): Optional. Number of shared-expert ranks.
    - Atlas A2 training products/Atlas A2 inference products: Expert sharing is not supported. Retain the default value.
    - Atlas A3 training products/Atlas A3 inference products: The value range is [0, `ep_world_size`). If the value is not 0, the condition `shared_expert_rank_num % shared_expert_num == 0` must be satisfied.

- **`global_bs`** (`int`): Optional. Global batch size within the EP communication domain. When the batch sizes differ across ranks, passing `max_bs * ep_world_size` is supported, where `max_bs` indicates the maximum batch size of a single rank. When the batch sizes are identical across ranks, passing `0` or `BS * ep_world_size` is supported.

- **`comm_quant_mode`** (`int`): Optional. Communication quantization type.
    - Atlas A2 training products/Atlas A2 inference products: The value can be `0` or `2`. `0` disables quantization during communication, and `2` enables `int8` quantization during communication. The value `2` is supported only when `comm_alg` is set to `"hierarchy"`, or when `HCCL_INTRA_PCIE_ENABLE=1`, `HCCL_INTRA_ROCE_ENABLE=0`, and the driver version is not earlier than 25.0.RC1.1.
    - Atlas A3 training products/Atlas A3 inference products: The value can be `0` or `2`. `0` disables quantization during communication, and `2` enables `int8` quantization during communication. `int8` quantization can be enabled only when `tp_world_size` is not `2`.

- **`comm_alg`** (`str`): Optional. Communication-optimized memory-layout algorithm.
    - Atlas A2 training products/Atlas A2 inference products: The current version supports `""`, `"fullmesh"`, and `"hierarchy"`. You are advised to use `"hierarchy"` together with driver version 25.0.RC1.1 or later.
        - `""`: If `HCCL_INTRA_PCIE_ENABLE=1` and `HCCL_INTRA_ROCE_ENABLE=0`, the `"hierarchy"` algorithm is used. Otherwise, the `"fullmesh"` algorithm is used. This mode is not recommended.
        - `"fullmesh"`: Token data is sent directly back to the target rank through RDMA.
        - `"hierarchy"`: Token data is transmitted in two stages: intra-server and inter-server. The same token data is first aggregated and summed within a server, and then transmitted across servers to reduce the data volume of inter-server communication.
    - Atlas A3 training products/Atlas A3 inference products: The current version supports `""` and `"hierarchy"`.
        - `""`: Default value. Token data is sent directly back to the target rank through MTE.
        - `"hierarchy"`: Token data is transmitted in two stages: intra-server and inter-server. The same token data is first aggregated and summed within a server, and then transmitted across servers to reduce the data volume of inter-server communication. This template supports only scenarios where `tp_world_size` is 1 and the number of shared experts is 0, and does not support 2D masks, special experts, dynamic scale-in, or performance profiling.

- **`zero_expert_num`** (`int`): Optional. Number of zero experts.
    - Atlas A2 training products/Atlas A2 inference products:
         - When `comm_alg` is set to `"fullmesh"`, the value range is [0, MAX_INT32), where the value of `MAX_INT32` is `2147483647`. Valid zero-expert ID values are in the range [`moe_expert_num`, `moe_expert_num + zero_expert_num`). A non-zero value enables the zero-computation expert feature, which is currently experimental and must be used with caution.
         - When `comm_alg` is set to `"hierarchy"`, this parameter is currently not supported. Use the default value `None`.
    - Atlas A3 training products/Atlas A3 inference products: The value range is [0, `MAX_INT32`), where the value of `MAX_INT32` is `2147483647`. Valid zero-expert ID values are in the range [`moe_expert_num`, `moe_expert_num + zero_expert_num`).

- **`copy_expert_num`** (`int`): Optional. Number of copy experts.
    - Atlas A2 training products/Atlas A2 inference products:
         - When `comm_alg` is set to `"fullmesh"`, the value range is [0, `MAX_INT32`), where the value of `MAX_INT32` is `2147483647`. Valid copy-expert ID values are in the range [`moe_expert_num + zero_expert_num`, `moe_expert_num + zero_expert_num + copy_expert_num`). A non-zero value enables the zero-computation expert feature, which is currently experimental and must be used with caution.
         - When `comm_alg` is set to `"hierarchy"`, this parameter is currently not supported. Use the default value `None`.
    - Atlas A3 training products/Atlas A3 inference products: The value range is [0, `MAX_INT32`), where the value of `MAX_INT32` is `2147483647`. Valid copy-expert ID values are in the range [`moe_expert_num + zero_expert_num`, `moe_expert_num + zero_expert_num + copy_expert_num`).

- **`const_expert_num`** (`int`): Optional. Number of constant experts.
    - Atlas A2 training products/Atlas A2 inference products: This parameter is currently not supported. Set it to `0`.
    - Atlas A3 training products/Atlas A3 inference products: The value range is [0, `MAX_INT32`), where the value of `MAX_INT32` is `2147483647`. Valid constant-expert ID values are in the range [`moe_expert_num + zero_expert_num + copy_expert_num`, `moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num`).

## Return Values<a name="en-us_topic_0000002168254826_section22231435517"></a>

`Tensor`

Processed tokens. This parameter must be 2D with shape `(BS, H)`. The data type can be `bfloat16` or `float16`. The data type must be identical to that of the input `expand_x`. The data layout can be ND. Non-contiguous tensors are not supported.

## Constraints<a name="en-us_topic_0000002168254826_section12345537164214"></a>

- This API can be used in inference scenarios.
- This API supports graph mode. `npu_moe_distribute_dispatch_v2` and `npu_moe_distribute_combine_v2` must be used together.
- Element values inside the `assist_info_for_combine`, `ep_recv_counts`, `tp_recv_counts`, and `expand_scales` tensors output by `npu_moe_distribute_dispatch_v2` can vary across different product models, communication algorithms, or software versions. When using this API, pass these tensors directly to the corresponding parameters of `npu_moe_distribute_combine_v2`. Other service logic of the model must not depend on them.
- Values of the `group_ep`, `ep_world_size`, `moe_expert_num`, `group_tp`, `tp_world_size`, `expert_shard_type`, `shared_expert_num`, `shared_expert_rank_num`, and `global_bs` parameters must remain identical across all ranks during the API execution process. In addition, the values of `group_ep`, `ep_world_size`, `group_tp`, `tp_world_size`, `expert_shard_type`, and `global_bs` must also remain identical across different layers in the network and must match the corresponding parameters in [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md).
- Atlas A3 training products/Atlas A3 inference products: In this scenario, a single rank contains dual dies. Therefore, "this rank" in the parameter description indicates a single die.
- `moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num < MAX_INT32`
- Variables used in parameter tensor shapes:
    - `A`: Maximum number of tokens that can be received by the current rank. The value range is as follows:
        - For shared experts: `A = BS * ep_world_size * shared_expert_num/shared_expert_rank_num`.
        - For MoE experts, when `global_bs` is `0`: `A >= BS * ep_world_size * min(local_expert_num, K)`. When `global_bs` is not `0`: `A >= global_bs * min(local_expert_num, K)`.

    - `H`: Hidden layer size.
        - Atlas A2 training products/Atlas A2 inference products: The value range of `H` is as follows:
            - When `comm_alg` is set to `"fullmesh"`, the value range of `H` is (0, 7168], and `H` must be a multiple of 32.
            - When `comm_alg` is set to `"hierarchy"` and the driver version is not earlier than 25.0.RC1.1, the value range of `H` is (0, 10 * 1024], and `H` must be a multiple of 32.
        - Atlas A3 training products/Atlas A3 inference products: The value range is [1024, 8192].

    - `BS`: Number of tokens to be sent.
        - Atlas A2 training products/Atlas A2 inference products: The value range of `BS` is as follows:
            - When `comm_alg` is set to `"fullmesh"`: 0 < `BS` <= 256.
            - When `comm_alg` is set to `"hierarchy"` and the Ascend HDK version is not earlier than 25.0.RC1.1: 0 < `BS` <= 512.
        - Atlas A3 training products/Atlas A3 inference products: The value range of `BS` is as follows.
            - When `comm_alg` is set to `""`: 0 < `BS` <= 512.
            - When `comm_alg` is set to `"hierarchy"`: 0 < `BS` <= 256.

    - `K`: Number of top-K experts selected. The value range is 0 < `K` <= 16, and the condition 0 < `K` <= `moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num` must be satisfied.

    - `server_num`: Number of server nodes. Valid values are `2`, `4`, or `8`.
        - Atlas A2 training products/Atlas A2 inference products: This variable is used only in shapes for this scenario.

    - `local_expert_num`: Number of experts on the current rank.
        - For shared-expert ranks, `local_expert_num = 1`.
        - For MoE expert ranks, `local_expert_num = moe_expert_num/(ep_world_size - shared_expert_rank_num)`. When `local_expert_num > 1`, communication in the TP domain is not supported.
        - Atlas A3 training products/Atlas A3 inference products: The condition `0 < local_expert_num * ep_world_size <= 2048` must be satisfied.

- HCCL communication domain buffer size:

    Before calling this API, verify that the configured HCCL communication domain buffer size is reasonable. The unit is MB, and the default value is `200` MB if not configured.
    - Atlas A2 training products/Atlas A2 inference products:
        The buffer size can be configured using the `HCCL_BUFFSIZE` environment variable.
        - When `comm_alg` is set to `""`: `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE` take effect only under this configuration. The `"fullmesh"` or `"hierarchy"` formula is selected based on the `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE` configurations.
        - When `comm_alg` is set to `"fullmesh"`: The configured size must be greater than or equal to `2 * (BS * ep_world_size * min(local_expert_num, K) * H * sizeof(uint16) + 2MB)`.
        - When `comm_alg` is set to `"hierarchy"`: The configured size must be greater than or equal to `(moe_expert_num + ep_world_size/4) * Align512(max_bs * (H * sizeof(dtype_x) + 4 * Align8(K) * sizeof(uint32))) * 1B + 8MB`, where `Align512(x) = ((x + 512 - 1)/512) * 512` and `Align8(x) = ((x + 8 - 1)/8) * 8`.

    - Atlas A3 training products/Atlas A3 inference products:
        The buffer size can be configured through either the `HCCL_BUFFSIZE` environment variable or the `hccl_buffer_size` parameter. For details, see section "hccl_buffer_size" in [PyTorch Training Model Porting and Tuning](https://hiascend.com/document/redirect/canncommercial-ptmigr) (path: **Performance Profiling** > **Performance Profiling Methods** > **Communication Optimization** > **Optimization Methods** > **hccl_buffer_size**).
        - Within the EP communication domain: The configured size must be greater than or equal to 2, and it must satisfy the condition `>= 2 * (local_expert_num * max_bs * ep_world_size * Align512(Align32(2 * h) + 64) + (K + shared_expert_num) * max_bs * Align512(2 * h))`.
        - Within the TP communication domain: The value must be greater than or equal to `(A * Align512(Align32(h * 2) + 44) + A * Align512(h * 2)) * 2`.
        - The alignment functions are defined as follows: $Align480(x) = ((x + 480 - 1)/480) * 512$, $Align512(x) = ((x + 512 - 1)/512) * 512$, $Align32(x) = ((x + 32 - 1)/32) * 32$.
        - When `comm_alg` is set to `"hierarchy"`, only the `HCCL_BUFFSIZE` environment variable is supported for configuration.

- `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE`:
    - Atlas A2 training products/Atlas A2 inference products: These environment variables are no longer recommended. You are advised to set `comm_alg` to `"hierarchy"`.

- In formulas in this document, the division sign (/) indicates integer division.

- Communication domain usage constraints:

    - The `npu_moe_distribute_dispatch_v2` and `npu_moe_distribute_combine_v2` operators within a single model must operate in the same EP communication domain, which must not include other operators.

    - The `npu_moe_distribute_dispatch_v2` and `npu_moe_distribute_combine_v2` operators within a single model must either operate in the same TP communication domain or both operate without a TP communication domain. When a TP communication domain is involved, this domain must not include other operators.

- Networking constraints:
    - Atlas A2 training products/Atlas A2 inference products: In multi-server deployments, only switch-based networking is supported. Direct connections between two servers are not supported.

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
            ep_rank_id=rank//tp_world_size,
            tp_rank_id=rank%tp_world_size,
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
