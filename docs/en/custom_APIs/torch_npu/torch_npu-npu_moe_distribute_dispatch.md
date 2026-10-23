# torch_npu.npu_moe_distribute_dispatch

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

## Function<a name="en-us_topic_0000002203575833_section14441124184110"></a>

- Performs quantization on token data (optional), followed by `alltoallv` communication in the Expert Parallelism (EP) domain, and then `allgatherv` communication in the Tensor Parallelism (TP) domain (optional). This API must be used together with [torch_npu.npu_moe_distribute_combine](torch_npu-npu_moe_distribute_combine.md) to implement token dispatch and combine for MoE parallel deployment.

- Formulas: ($x$ indicates the input `x`, $scales$ indicates the input `scales`, and $quant\_mode$ indicates the input `quant_mode`.)
    - When `quant_mode` is not `2` (non-dynamic quantization scenario):

        $$
        quant\_out =
        \begin{cases}
        x, & quant\_mode=0 \\
        CastToInt8(CastToFp32(x) \times scales), & quant\_mode \ne 0
        \end{cases}
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

    - When `quant_mode` is set to 2 (dynamic quantization scenario):

        $$
        x\_fp32 = CastToFp32(x) \times scales
        $$

        $$
        dynamic\_scales\_value = 127.0 / Max(Abs(x\_fp32))
        $$

        $$
        quant\_out = CastToInt8(x\_fp32 \times dynamic\_scales\_value)
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

## Prototype<a name="en-us_topic_0000002203575833_section45077510411"></a>

```python
torch_npu.npu_moe_distribute_dispatch(x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, *, scales=None, x_active_mask=None, expert_scales=None, group_tp="", tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, quant_mode=0, global_bs=0, expert_token_nums_type=1) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## Parameters<a name="en-us_topic_0000002203575833_section112637109429"></a>

- **`x`** (`Tensor`): Token data used for computation and dispatched to other ranks based on `expert_ids`. This parameter must be 2D with shape `(BS, H)`, indicating that there are `BS` (batch size) tokens. The data type can be `torch.bfloat16` or `torch.float16`. The data layout is ND. Non-contiguous tensors are supported.
- **`expert_ids`** (`Tensor`): Top-K expert indices for each token, determining the experts to which each token is dispatched. This parameter must be 2D with shape `(BS, K)`. The data type can be `torch.int32`. The data layout is ND. Non-contiguous tensors are supported. This parameter corresponds to the `expert_ids` input of [torch_npu.npu_moe_distribute_combine](torch_npu-npu_moe_distribute_combine.md). The value range of elements inside the tensor is [0, `moe_expert_num`), and the `K` values within the same row must be unique.
- **`group_ep`** (`str`): EP communication domain name used for expert parallelism. The string length range is [1, 128). The value must differ from `group_tp`.
- **`ep_world_size`** (`int`): Size of the EP communication domain.

    <!-- npu="910b" id4 -->
    - <term>Atlas A2 products</term>: Valid values are `16`, `32`, or `64`.
    <!-- end id4 -->
    <!-- npu="A3" id5 -->
    - <term>Atlas A3 products</term>: Valid values are `8`, `16`, `32`, `64`, `128`, `144`, `256`, or `288`.
    <!-- end id5 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950DT products</term>: The supported values are 2, 4, 8, 16, 32, 64, 128, 144, 256, and 288.
    <!-- end id6 -->

- **`ep_rank_id`** (`int`): Rank ID of the current rank within the EP communication domain. The value range is [0, `ep_world_size`). The `ep_rank_id` values of all ranks within the same EP communication domain must be unique.
- **`moe_expert_num`** (`int`): Number of MoE experts. The value range is [1, 512], and the condition `moe_expert_num % (ep_world_size - shared_expert_rank_num) == 0` must be satisfied.

    <!-- npu="910b" id7 -->
    - <term>Atlas A2 products</term>: The following condition must also be met: `moe_expert_num/(ep_world_size - shared_expert_rank_num) <= 24`.
    <!-- end id7 -->

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`scales`** (`Tensor`): Optional. Weights of each expert. This parameter is omitted in non-quantization scenarios and is optional in dynamic quantization scenarios. If provided, it must be 2D. When shared experts are present, the shape is `(shared_expert_num+moe_expert_num, H)`; when no shared experts are present, the shape is `(moe_expert_num, H)`. The data layout is ND. Non-contiguous tensors are not supported.

    <!-- npu="910b" id8 -->
    - <term>Atlas A2 products</term>: The data type supports `torch.float32`. When `HCCL_INTRA_PCIE_ENABLE` is 1 and `HCCL_INTRA_ROCE_ENABLE` is 0, the default value must be passed.
    <!-- end id8 -->
    <!-- npu="950,A3" id9 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The data type supports `torch.float32`. When `quant_mode` is 0, this parameter does not need to be passed. When `quant_mode` is 2, valid data or null can be passed. When valid data is passed, the shape is `(H,)`, `(shared_expert_num+moe_expert_num, H)`, or `(moe_expert_num, H)`.
    <!-- end id9 -->

- **`x_active_mask`** (`Tensor`): Optional. Specifies whether the tokens in `x` are valid. A value of `false` indicates invalid.

    <!-- npu="910b" id10 -->
    - <term>Atlas A2 products</term>: Reserved parameter, not supported in the current version. Use the default value.
    <!-- end id10 -->
    <!-- npu="A3" id11 -->
    - <term>Atlas A3 products</term>: This parameter must be 1D with shape `(BS,)`. The data type can be `torch.bool`. The data layout is ND. Non-contiguous tensors are supported. By default, all tokens are valid. When the BS values differ across ranks, all tokens must be valid.
    <!-- end id11 -->
    <!-- npu="950" id12 -->
    - <term>Ascend 950DT products</term>: Reserved parameter, not supported in the current version. Use the default value.
    <!-- end id12 -->

- **`expert_scales`** (`Tensor`): Optional. Top-K expert weights for each token.

    <!-- npu="910b" id13 -->
    - <term>Atlas A2 products</term>: This parameter must be 2D with shape `(BS, K)`. The data type can be `torch.float`. The data layout is ND. Non-contiguous tensors are supported.
    <!-- end id13 -->
    <!-- npu="950,A3" id14 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: Reserved parameter, not supported in the current version. Use the default value.
    <!-- end id14 -->

- **`group_tp`** (`str`): Optional. TP communication domain name, that is, the communication domain for data parallelism. This parameter is required if TP domain communication exists. If there is no TP domain communication, use the default value.

    <!-- npu="910b" id15 -->
    - <term>Atlas A2 products</term>: TP domain communication is not supported. Use the default value.
    <!-- end id15 -->
    <!-- npu="A3" id16 -->
    - <term>Atlas A3 products</term>: When TP domain communication is involved, the string length range is [1, 128), and the value must differ from `group_ep`.
    <!-- end id16 -->
    <!-- npu="950" id17 -->
    - <term>Ascend 950DT products</term>: Reserved parameter, not supported in the current version. Use the default value.
    <!-- end id17 -->

- **`tp_world_size`** (`int`): Optional. Size of the TP communication domain. This parameter is required only when TP domain communication is involved.

    <!-- npu="910b" id18 -->
    - <term>Atlas A2 products</term>: TP domain communication is not supported. Use the default value `0`.
    <!-- end id18 -->
    <!-- npu="A3" id19 -->
    - <term>Atlas A3 products</term>: When TP domain communication is involved, the value range is [0, 2]. The values `0` and `1` indicate that there is no TP domain communication, and the value `2` indicates that there is TP domain communication.
    <!-- end id19 -->
    <!-- npu="950" id20 -->
    - <term>Ascend 950DT products</term>: In the current version, only 1 can be passed.
    <!-- end id20 -->

- **`tp_rank_id`** (`int`): Optional. Rank ID of the current rank within the TP communication domain. This parameter is required only when TP domain communication is involved.

    <!-- npu="910b" id21 -->
    - <term>Atlas A2 products</term>: TP domain communication is not supported. Use the default value.
    <!-- end id21 -->
    <!-- npu="A3" id22 -->
    - <term>Atlas A3 products</term>: When TP domain communication is involved, the value range is [0, 1]. The default value is `0`, and the `tp_rank_id` values of all ranks within the same TP communication domain must be unique. If there is no TP domain communication, pass `0`.
    <!-- end id22 -->
    <!-- npu="950" id23 -->
    - <term>Ascend 950DT products</term>: TP domain communication is not supported. Pass 0.
    <!-- end id23 -->

- **`expert_shard_type`** (`int`): Optional. Layout type of shared-expert ranks.

    <!-- npu="910b" id24 -->
    - <term>Atlas A2 products</term>: This parameter is not supported currently. Use the default value.
    <!-- end id24 -->
    <!-- npu="950,A3" id25 -->
    - <term>Atlas A3 products</term> and <term>Ascend 950DT products</term>: Currently, only `0` is supported, indicating that shared-expert ranks are placed before MoE expert ranks.
    <!-- end id25 -->

- **`shared_expert_num`** (`int`): Optional. Number of shared experts, where a shared expert can be replicated and deployed across multiple ranks.

    <!-- npu="910b" id26 -->
    - <term>Atlas A2 products</term>: This parameter is not supported currently. Use the default value.
    <!-- end id26 -->
    <!-- npu="A3" id27 -->
    - <term>Atlas A3 products</term>: The value range is [0, 4]. A value of `0` indicates that there is no shared expert, and the default value is 1.
    <!-- end id27 -->
    <!-- npu="950" id28 -->
    - <term>Ascend 950DT products</term>: In the current version, only 1 is supported, where 1 indicates one shared expert.
    <!-- end id28 -->

- **`shared_expert_rank_num`** (`int`): Optional. Number of shared-expert ranks.

    <!-- npu="910b" id29 -->
    - <term>Atlas A2 products</term>: Shared experts are not supported. Set it to `0`.
    <!-- end id29 -->
    <!-- npu="950,A3" id30 -->
    - <term>Atlas A3 products</term> and <term>Ascend 950DT products</term>: The value range is [0, ep_world_size). A value of `0` indicates that there is no shared expert; a non-zero value requires `ep_world_size % shared_expert_rank_num = 0`.
    <!-- end id30 -->

- **`quant_mode`** (`int`): Optional. Quantization mode. Valid values: `0` indicates non-quantization (default), and `2` indicates dynamic quantization. When `quant_mode` is `2`, `dynamic_scales` is not `None`; when `quant_mode` is `0`, `dynamic_scales` is `None`.
- **`global_bs`** (`int`): Optional. Global batch size of the EP domain.

    <!-- npu="910b" id31 -->
    - <term>Atlas A2 products</term>: When the batch sizes differ across ranks, passing `max_bs * ep_world_size` or `256 * ep_world_size` is supported, where `max_bs` indicates the maximum batch size of a single rank. Passing `max_bs * ep_world_size` is recommended. Consistently passing `256 * ep_world_size` will result in a lack of support in future versions for scenarios where the batch size is greater than 256. When the batch sizes are identical across ranks, passing `0` or `BS * ep_world_size` is supported.
    <!-- end id31 -->
    <!-- npu="950,A3" id32 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: When the BS values of each rank are different, `max_bs*ep_world_size` can be passed, where `max_bs` indicates the maximum BS of a single rank. When the BS values of each rank are the same, the value can be 0 or `BS*ep_world_size`.
    <!-- end id32 -->

- **`expert_token_nums_type`** (`int`): Optional. Value type of the output `expert_token_nums`. The value range is [0, 1]. `0` indicates the prefix sum of the numbers of tokens received by each expert, and `1` indicates the number of tokens received by each expert (default).

## Return Values

- **`expand_x`** (`Tensor`): Token data received by the current rank. This parameter must be 2D with shape `(max(tp_world_size, 1) * A, H)`, where A indicates the maximum number of tokens that can be received within the EP communication domain. The data layout is ND. Non-contiguous tensors are supported. In quantization scenarios, the data type is `torch.int8`; in non-quantization scenarios, it is the same as the data type of `x`.

    <!-- npu="950,A3,910b" id33 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The data type can be `torch.bfloat16`, `torch.float16`, or `torch.int8`.
    <!-- end id33 -->

- **`dynamic_scales`** (`Tensor`): Dynamically quantized parameters obtained through computation. This output is available only when `quant_mode` is not `0`. This parameter must be 1D with shape `(A,)`. The data type can be `torch.float`. The data layout can be ND. Non-contiguous tensors are supported.
- **`expand_idx`** (`Tensor`): Number of tokens sent to the same expert. This parameter must be 1D with shape `(BS * K,)`. The data type can be `torch.int32`. The data layout is ND. Non-contiguous tensors are supported. This parameter corresponds to the `expand_idx` input of [torch_npu.npu_moe_distribute_combine](torch_npu-npu_moe_distribute_combine.md).
- **`expert_token_nums`** (`Tensor`): Number of tokens received by each expert on the current rank. It must be 1D with shape `(local_expert_num,)`. The data type can be `torch.int64`. The data layout can be ND. Non-contiguous tensors are supported.
- **`ep_recv_counts`** (`Tensor`): Number of tokens (represented as a prefix sum) received by each rank in the EP communication domain. This parameter must be 1D. The data type is `torch.int32`. The data layout can be ND. Non-contiguous tensors are supported. This parameter corresponds to the `ep_send_counts` input of [torch_npu.npu_moe_distribute_combine](torch_npu-npu_moe_distribute_combine.md).

    <!-- npu="910b" id34 -->
    - <term>Atlas A2 products</term>: The shape must be `(moe_expert_num + 2 * global_bs * K * server_num,)`. The first `moe_expert_num` elements indicate the token counts received by each expert on the current rank from other ranks within the EP communication domain, represented as prefix sums. The remaining `2 * global_bs * K * server_num` elements are used to store the token counts that can be reduced in advance during the combine operation and the communication buffer offsets before executing inter-server and intra-server communication. When the value of `global_bs` is `0`, the value is calculated as `bs * ep_world_size`.
    <!-- end id34 -->
    <!-- npu="950,A3" id35 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The shape must be (ep_world_size*max(tp_world_size, 1)*local_expert_num,).
    <!-- end id35 -->

- **`tp_recv_counts`** (`Tensor`): Number of tokens received by each rank in the TP communication domain. This parameter corresponds to the `tp_send_counts` input of [torch_npu.npu_moe_distribute_combine](torch_npu-npu_moe_distribute_combine.md).

    <!-- npu="950,910b" id36 -->
    - <term>Atlas A2 products</term>, <term>Ascend 950DT products</term>: The TP communication domain is not supported, and this output is not available.
    <!-- end id36 -->
    <!-- npu="A3" id37 -->
    - <term>Atlas A3 products</term>: The TP communication domain is supported. This parameter must be 1D with shape `(tp_world_size,)`. The data type can be `torch.int32`. The data layout is ND. Non-contiguous tensors are supported.
    <!-- end id37 -->

- **`expand_scales`** (`Tensor`): Output after the alltoallv operation is performed on `expert_scales` and `x`.

    <!-- npu="910b" id38 -->
    - <term>Atlas A2 products</term>: This parameter must be 1D with shape `(A,)`. The data type can be `torch.float`. The data layout is ND. Non-contiguous tensors are supported.
    <!-- end id38 -->
    <!-- npu="950,A3" id39 -->
    - <term>Atlas A3 products</term> and <term>Ascend 950DT products</term>: This output is not supported. `None` is returned.
    <!-- end id39 -->

## Constraints

- This API can be used in inference scenarios.
<!-- npu="950" id40 -->
- Communication mode constraints:
    - <term>Ascend 950DT products</term>: Only UB Memory communication is supported.
<!-- end id40 -->

- This API supports single-operator mode and static graph mode, and `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` must be used in combination.
- Element values inside the `expand_idx`, `ep_recv_counts`, `tp_recv_counts`, and `expand_scales` tensors output by `npu_moe_distribute_dispatch` can vary across different product models, communication algorithms, or software versions. When using this API, pass these tensors directly to the corresponding parameters of `npu_moe_distribute_combine`. Other service logic of the model must not depend on them.
- The values of the `group_ep`, `ep_world_size`, `moe_expert_num`, `group_tp`, `tp_world_size`, `expert_shard_type`, `shared_expert_num`, `shared_expert_rank_num`, and `global_bs` parameters must remain identical across all ranks during the API call process. In addition, the values of `group_ep`, `ep_world_size`, `moe_expert_num`, `group_tp`, `tp_world_size`, `expert_shard_type`, and `global_bs` must also remain identical across different layers in the network and must match the corresponding parameters in [torch_npu.npu_moe_distribute_combine](torch_npu-npu_moe_distribute_combine.md).

<!-- npu="A3" id41 -->
- <term>Atlas A3 products</term>: Only the value `7168` is supported.
<!-- end id41 -->
- Variables used in the shapes of parameters are as follows:
    - `BS`: Number of tokens to be sent.
        - For shared experts, when `global_bs` is `0`, `A=BS*shared_expert_num/shared_expert_rank_num` must be satisfied; when `global_bs` is not `0`, `A=global_bs*shared_expert_num/shared_expert_rank_num` must be satisfied.
        - For MoE experts, when `global_bs` is `0`, `A>=BS*ep_world_size*min(local_expert_num, K)` must be satisfied; when `global_bs` is not `0`, `A>=global_bs*min(local_expert_num, K)` must be satisfied.

    - H: hidden size.

        <!-- npu="910b" id42 -->
        - <term>Atlas A2 products</term>: The value range is (0, 7168], and the value must be an integer multiple of 32.
        <!-- end id42 -->
        <!-- npu="A3" id43 -->
        - <term>Atlas A3 products</term>: The value is in the range [1024, 7168].
        <!-- end id43 -->
        <!-- npu="950" id44 -->
        - <term>Ascend 950DT products</term>: The value is 7168.
        <!-- end id44 -->

    - BS: Number of tokens to be sent.

        <!-- npu="910b" id45 -->
        - <term>Atlas A2 products</term>: The value range is 0 < BS <= 256.
        <!-- end id45 -->
        <!-- npu="950,A3" id46 -->
        - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The value range is 0 < BS <= 512.
        <!-- end id46 -->

    - K: number of top-K experts selected, and 0 < K <= moe_expert_num must be satisfied.

        <!-- npu="910b" id47 -->
        - <term>Atlas A2 products</term>: The value range must be 0 < K <= 16.
        <!-- end id47 -->
        <!-- npu="A3" id48 -->
        - <term>Atlas A3 products</term>: The value must be in the range 0 < K ≤ 8.
        <!-- end id48 -->
        <!-- npu="950" id49 -->
        - <term>Ascend 950DT products</term>: The value range is 0 < K <= 8 and less than moe_expert_num.
        <!-- end id49 -->

    - server_num: number of server nodes. In single-server scenarios, the value can be 1; in multi-server scenarios, only the values 2, 4, and 8 are supported.

        <!-- npu="910b" id50 -->
        - <term>Atlas A2 products</term>: This variable is used only in the shapes of this scenario.
        <!-- end id50 -->

    - local_expert_num: Number of experts on the current rank.
        - For shared-expert ranks, `local_expert_num = 1`.
        - For MoE expert ranks, `local_expert_num = moe_expert_num/(ep_world_size - shared_expert_rank_num)`. When `local_expert_num > 1`, communication in the TP domain is not supported.

- HCCL communication domain buffer size:

    Before calling this API, verify that the configured HCCL_BUFFSIZE environment variable is reasonable.

    > [!NOTE]
    > CANN environment variable HCCL_BUFFSIZE: memory size occupied by a single communication domain, in MB. The default value is 200 MB if not configured.

    <!-- npu="910b" id51 -->
    - <term>Atlas A2 products</term>: The value must be >= 2*(BS*ep_world_size*min(local_expert_num, K)*H*sizeof(`torch.uint16`)+2MB).
    <!-- end id51 -->
    <!-- npu="A3" id52 -->
    - <term>Atlas A3 products</term>: The value must be >=2 and must satisfy >= 2*(local_expert_num*max_bs*ep_world_size*Align512(Align32(2*H)+64)+(K+shared_expert_num)*max_bs*Align512(2*H)), where local_expert_num must be the number of experts on the current MoE expert rank.
    <!-- end id52 -->
    <!-- npu="950" id53 -->
    - <term>Ascend 950DT products</term>: The value must be >=2 and must satisfy 1024^2*(HCCL_BUFFSIZE-2)/2 >= BS*2*(H+128)*(ep_world_size*local_expert_num+K+1), where local_expert_num must be the number of experts on the current MoE expert rank.
    <!-- end id53 -->

<!-- npu="910b" id54 -->
- <term>Atlas A2 products</term>: Configuring the environment variables `HCCL_INTRA_PCIE_ENABLE=1` and `HCCL_INTRA_ROCE_ENABLE=0` can reduce the cross-server communication data volume and improve operator performance. In this case, the condition `HCCL_BUFFSIZE >= moe_expert_num * BS * (H * sizeof(dtypeX) + 4 * ((K + 7) / 8 * 8) * sizeof(`torch.uint32`)) + 4MB + 100MB` must be satisfied. Furthermore, for the input parameter `moe_expert_num`, only the condition `moe_expert_num % (ep_world_size - shared_expert_rank_num) == 0` must be satisfied, and the condition `moe_expert_num / (ep_world_size - shared_expert_rank_num) <= 24` is not required.
<!-- end id54 -->
- Communication domain usage constraints:
    - In a model, the `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` operators support only the same EP communication domain, and no other operator is allowed in the domain.
    - In a model, the `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` operators support only the same TP communication domain, or neither supports TP communication. When a TP communication domain exists, no other operator is allowed in the domain.

    <!-- npu="A3" id55 -->
    - <term>Atlas A3 products</term>: Nodes in a communication domain must reside within the same SuperPoD. Cross-SuperPoD deployment is not supported.
    <!-- end id55 -->

<!-- npu="910b" id56 -->
- Networking constraints:
    - <term>Atlas A2 products</term>: In multi-server deployments, only switch-based networking is supported. Direct server-to-server connections are not supported.
<!-- end id56 -->

- Version compatibility constraints:

    In static graph mode, starting with TorchNPU 7.1.0, the TorchNPU framework performs strict validation between the Meta inference results and inferShape inference results of the output of the last node in the static graph. If the graph contains only one Dispatch operator and the CANN version is earlier than the TorchNPU version, a shape mismatch error may occur. You are advised to upgrade the CANN version. For details about version compatibility information, see section **Related Product Versions** in the [TorchNPU Release Notes](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md).

- In the formula, "/" indicates integer division.

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
    quant_mode = 2                       # 2 indicates dynamic quantization
    is_dispatch_scales = True            # For dynamic quantization, you can choose whether to pass scales
    input_dtype = torch.bfloat16         # Output data type
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # Number of dies per host
    sharedExpertRankNum = 2                      # Number of shared experts
    moeExpertNum = 14                            # Number of MOE experts
    bs = 8                                       # Number of tokens
    h = 7168                                     # Length of each token
    k = 8
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - sharedExpertRankNum
    local_moe_expert_num = moeExpertNum // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (sharedExpertRankNum > 0)
    is_quant = (quant_mode > 0)

    def gen_unique_topk_array(low, high, bs, k):
        array = []
        for i in range(bs):
            top_idx = list(np.arange(low, high, dtype=np.int32))
            random.shuffle(top_idx)
            array.append(top_idx[0:k])
        return np.array(array)

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

    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        ep_group, tp_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        tp_hcomm_info = get_hcomm_info(rank, tp_group)

        # Create input tensors
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = gen_unique_topk_array(0, moeExpertNum, bs, k).astype(np.int32)
        expert_ids = torch.from_numpy(expert_ids).npu()

        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + moeExpertNum, h) if sharedExpertRankNum else (moeExpertNum, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None

        expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales = torch_npu.npu_moe_distribute_dispatch(
            x=x,
            expert_ids=expert_ids,
            group_ep=ep_hcomm_info,
            group_tp=tp_hcomm_info,
            ep_world_size=ep_world_size,
            tp_world_size=tp_world_size,
            ep_rank_id=rank // tp_world_size,
            tp_rank_id=rank % tp_world_size,
            expert_shard_type=0,
            shared_expert_rank_num=sharedExpertRankNum,
            moe_expert_num=moeExpertNum,
            scales=scales,
            quant_mode=quant_mode,
            global_bs=globalBS)
        if is_quant:
            expand_x = expand_x.to(input_dtype)
        x = torch_npu.npu_moe_distribute_combine(expand_x=expand_x,
                                                 expert_ids=expert_ids,
                                                 expand_idx=expand_idx,
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
                                                 shared_expert_rank_num=sharedExpertRankNum,
                                                 moe_expert_num=moeExpertNum,
                                                 global_bs=globalBS)
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')

    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"global_bs={globalBS}")
        print(f"shared_expert_rank_num={sharedExpertRankNum}")
        print(f"moe_expert_num={moeExpertNum}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
        print(f"tp_world_size={tp_world_size}", flush=True)
        print(f"ep_world_size={ep_world_size}", flush=True)

        if tp_world_size != 1 and local_moe_expert_num > 1:
            print("unSupported tp = 2 and local moe > 1")
            exit(0)

        if sharedExpertRankNum > ep_world_size:
            print("sharedExpertRankNum cannot be greater than ep_world_size")
            exit(0)

        if sharedExpertRankNum > 0 and ep_world_size % sharedExpertRankNum != 0:
            print("ep_world_size must be an integer multiple of sharedExpertRankNum")
            exit(0)

        if moeExpertNum % moe_rank_num != 0:
            print("moeExpertNum must be an integer multiple of moe_rank_num")
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

    # Control mode
    quant_mode = 2                         # 2 indicates dynamic quantization
    is_dispatch_scales = True              # For dynamic quantization, you can choose whether to pass scales
    input_dtype = torch.bfloat16           # Output data type
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # Number of dies per host
    sharedExpertRankNum = 2                      # Number of shared experts
    moeExpertNum = 14                            # Number of MOE experts
    bs = 8                                       # Number of tokens
    h = 7168                                     # Length of each token
    k = 8
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - sharedExpertRankNum
    local_moe_expert_num = moeExpertNum // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (sharedExpertRankNum > 0)
    is_quant = (quant_mode > 0)

    class MOE_DISTRIBUTE_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, expert_ids, group_ep, group_tp, ep_world_size, tp_world_size,
                    ep_rank_id, tp_rank_id, expert_shard_type, shared_expert_rank_num, moe_expert_num,
                    scales, quant_mode, global_bs, expert_scales):
            output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch(x=x,
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
                                                                        global_bs=global_bs)

            expand_x_npu, _, expand_idx_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
            if expand_x_npu.dtype == torch.int8:
                expand_x_npu = expand_x_npu.to(input_dtype)
            output_combine_npu = torch_npu.npu_moe_distribute_combine(expand_x=expand_x_npu,
                                                                      expert_ids=expert_ids,
                                                                      expand_idx=expand_idx_npu,
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
                                                                      global_bs=global_bs)
            x = output_combine_npu
            x_combine_res = output_combine_npu
            return [x_combine_res, output_combine_npu]

    def gen_unique_topk_array(low, high, bs, k):
        array = []
        for i in range(bs):
            top_idx = list(np.arange(low, high, dtype=np.int32))
            random.shuffle(top_idx)
            array.append(top_idx[0:k])
        return np.array(array)

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

    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        ep_group, tp_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        tp_hcomm_info = get_hcomm_info(rank, tp_group)

        # Create input tensors
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = gen_unique_topk_array(0, moeExpertNum, bs, k).astype(np.int32)
        expert_ids = torch.from_numpy(expert_ids).npu()

        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + moeExpertNum, h) if sharedExpertRankNum else (moeExpertNum, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None

        model = MOE_DISTRIBUTE_GRAPH_Model()
        model = model.npu()
        npu_backend = torchair.get_npu_backend()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        output = model.forward(x, expert_ids, ep_hcomm_info, tp_hcomm_info, ep_world_size, tp_world_size,
                               rank // tp_world_size, rank % tp_world_size, 0, sharedExpertRankNum, moeExpertNum, scales,
                               quant_mode, globalBS, expert_scales)
        torch.npu.synchronize()
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')

    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"global_bs={globalBS}")
        print(f"shared_expert_rank_num={sharedExpertRankNum}")
        print(f"moe_expert_num={moeExpertNum}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
        print(f"tp_world_size={tp_world_size}", flush=True)
        print(f"ep_world_size={ep_world_size}", flush=True)

        if tp_world_size != 1 and local_moe_expert_num > 1:
            print("unSupported tp = 2 and local moe > 1")
            exit(0)

        if sharedExpertRankNum > ep_world_size:
            print("sharedExpertRankNum cannot be greater than ep_world_size")
            exit(0)

        if sharedExpertRankNum > 0 and ep_world_size % sharedExpertRankNum != 0:
            print("ep_world_size must be an integer multiple of sharedExpertRankNum")
            exit(0)

        if moeExpertNum % moe_rank_num != 0:
            print("moeExpertNum must be an integer multiple of moe_rank_num")
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
