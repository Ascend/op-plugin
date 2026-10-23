# torch_npu.npu_moe_distribute_combine

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

- Fuses `reduce_scatterv` communication, `alltoallv` communication, and final data aggregation by multiplying the corresponding weights and summing the results. This API must be used together with [torch_npu.npu_moe_distribute_dispatch](torch_npu-npu_moe_distribute_dispatch.md), serving as the reverse data collection path of the `npu_moe_distribute_dispatch` operator.

- Formulas:

    $$
    rs\_out = ReduceScatterV(expand\_x)
    $$

    $$
    ata\_out = AlltoAllv(rs\_out)
    $$

    $$
    x = Sum(expert\_scales * ata\_out + expert\_scales * shared\_expert\_x)
    $$

## Prototype<a name="en-us_topic_0000002168254826_section45077510411"></a>

```python
torch_npu.npu_moe_distribute_combine(expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num, *, tp_send_counts=None, x_active_mask=None, activation_scale=None, weight_scale=None, group_list=None, expand_scales=None, shared_expert_x=None, group_tp="", tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, global_bs=0, out_dtype=0, comm_quant_mode=0, group_list_type=0) -> Tensor
```

## Parameters<a name="en-us_topic_0000002168254826_section112637109429"></a>

- **`expand_x`** (`Tensor`): Required. Expanded token features generated based on `expert_ids`. This parameter must be 2D with shape `(max(tp_world_size, 1) * A, H)`. The data type can be `torch.bfloat16` or `torch.float16`. The data layout is ND. Non-contiguous tensors are supported.
    <!-- npu="910b" id53 -->
    - Atlas A2 products do not support the shared expert scenario.
    <!-- end id53 -->

- **`expert_ids`** (`Tensor`): Required. Top-K expert indices for each token. This parameter must be 2D with shape `(BS, K)`. The data type can be `torch.int32`. The data layout is ND. Non-contiguous tensors are supported. This parameter corresponds to the `expert_ids` input of [torch_npu.npu_moe_distribute_dispatch](torch_npu-npu_moe_distribute_dispatch.md). The value range of elements inside the tensor is `[0, moe_expert_num)`, and the top-K values within the same row must be unique.
- **`expand_idx`** (`Tensor`): Required. Number of tokens dispatched to each expert. This parameter must be 1D with shape `(BS * K,)`. The data type can be `torch.int32`. The data layout is ND. Non-contiguous tensors are supported. This parameter corresponds to the `expand_idx` output of [torch_npu.npu_moe_distribute_dispatch](torch_npu-npu_moe_distribute_dispatch.md).
- **`ep_send_counts`** (`Tensor`): Required. Token count sent from each expert on the current rank to each rank within the Expert Parallelism (EP) communication domain, where token counts are represented as prefix sums. This parameter must be 1D. The data type can be `torch.int32`. The data layout is ND. Non-contiguous tensors are supported. This parameter corresponds to the `ep_recv_counts` output of [torch_npu.npu_moe_distribute_dispatch](torch_npu-npu_moe_distribute_dispatch.md).

    <!-- npu="910b" id4 -->
    - <term>Atlas A2 products</term>: The shape must be `(moe_expert_num + 2 * global_bs * K * server_num,)`. The first `moe_expert_num` elements indicate the token counts received by each expert on the current rank from other ranks within the EP communication domain, represented as prefix sums. The remaining `2 * global_bs * K * server_num` elements are used to store the token counts that can be reduced in advance during the combine operation and the communication buffer offsets before executing inter-server and intra-server communication. When the value of `global_bs` is `0`, the value is calculated as `bs * ep_world_size`.
    <!-- end id4 -->
    <!-- npu="950,A3" id5 -->
    - <term>Atlas A3 products</term> and <term>Ascend 950DT products</term>: The shape must be `(ep_world_size * max(tp_world_size, 1) * local_expert_num,)`.
    <!-- end id5 -->

- **`expert_scales`** (`Tensor`): Required. Weights of the top-K experts for each token. This parameter must be 2D with shape `(BS, K)`. Shared-expert configurations do not require a weight coefficient and are summed directly. The data type can be `torch.float`. The data layout is ND. Non-contiguous tensors are supported.
- **`group_ep`** (`str`): Required. EP communication domain name used for expert parallelism. The string length range is [1, 128). The value must differ from `group_tp`.
- **`ep_world_size`** (`int`): Required. Size of the EP communication domain.

    <!-- npu="910b" id6 -->
    - <term>Atlas A2 products</term>: The supported values are 16, 32, and 64.
    <!-- end id6 -->
    <!-- npu="A3" id7 -->
    - <term>Atlas A3 products</term>: The supported values are 8, 16, 32, 64, 128, 144, 256, and 288.
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950DT products</term>: The supported values are 2, 4, 8, 16, 32, 64, 128, 144, 256, and 288.
    <!-- end id8 -->

- **`ep_rank_id`** (`int`): Required. Rank ID of the current rank within the EP communication domain. The value range is [0, `ep_world_size`). The `ep_rank_id` values of all ranks within the same EP communication domain must be unique.
- **`moe_expert_num`** (`int`): Required. Number of MoE experts. The value range is [1, 512], and the condition `moe_expert_num % (ep_world_size - shared_expert_rank_num) == 0` must be satisfied.

    <!-- npu="910b" id9 -->
    - <term>Atlas A2 products</term>: The following condition must also be met: `moe_expert_num/(ep_world_size - shared_expert_rank_num) <= 24`.
    <!-- end id9 -->

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`tp_send_counts`** (`Tensor`): Optional. Data volume sent from each expert on the current rank to each rank within the Tensor Parallelism (TP) communication domain. This parameter corresponds to the `tp_recv_counts` output of [torch_npu.npu_moe_distribute_dispatch](torch_npu-npu_moe_distribute_dispatch.md).

    <!-- npu="950,910b" id10 -->
    - <term>Atlas A2 products</term>, <term>Ascend 950DT products</term>: The TP communication domain is not supported. Use the default input.
    <!-- end id10 -->
    <!-- npu="A3" id11 -->
    - <term>Atlas A3 products</term>: The TP communication domain is supported. This parameter must be 1D with shape `(tp_world_size,)`. The data type can be `torch.int32`. The data layout is ND. Non-contiguous tensors are supported.
    <!-- end id11 -->

- **`x_active_mask`** (`Tensor`): Optional. Specifies whether the tokens in the output `x` are valid. A value of `false` indicates that the token is invalid and is not received.

    <!-- npu="950,910b" id12 -->
    - <term>Atlas A2 products</term>, <term>Ascend 950DT products</term>: **Reserved parameter, currently not used. Retain the default value.**
    <!-- end id12 -->
    <!-- npu="A3" id13 -->
    - <term>Atlas A3 products</term>: This parameter must be 1D with shape `(BS,)`. The data type can be `torch.bool`. The data layout is ND. Non-contiguous tensors are supported. When the BS values differ across ranks, all tokens must be valid.
    <!-- end id13 -->

- **`activation_scale`** (`Tensor`): Optional. **Reserved parameter, currently not used. Retain the default value.**
- **`weight_scale`** (`Tensor`): Optional. **Reserved parameter, currently not used. Retain the default value.**
- **`group_list`** (`Tensor`): Optional. **Reserved parameter, currently not used. Retain the default value.**
- **`expand_scales`** (`Tensor`): Optional. This parameter corresponds to the `expand_scales` output of [torch_npu.npu_moe_distribute_dispatch](torch_npu-npu_moe_distribute_dispatch.md).

    <!-- npu="910b" id14 -->
    - <term>Atlas A2 products</term>: Required. This parameter must be 1D with shape `(A,)`. The data type can be `torch.float`. The data layout is ND. Non-contiguous tensors are supported.
    <!-- end id14 -->
    <!-- npu="950,A3" id15 -->
    - <term>Atlas A3 products</term> and <term>Ascend 950DT products</term>: This parameter is not supported currently. Use the default value.
    <!-- end id15 -->

- **`shared_expert_x`** (`Tensor`): Optional. The data type must be identical to that of `expand_x`. Use this parameter only when the number of shared-expert ranks `shared_expert_rank_num` is `0`. It indicates the shared-expert tokens whose values must be added after the combine operation.

    <!-- npu="950,910b" id16 -->
    - <term>Atlas A2 products</term>, <term>Ascend 950DT products</term>: This parameter is not supported currently. Use the default value.
    <!-- end id16 -->
    <!-- npu="A3" id17 -->
    - <term>Atlas A3 products</term>: This parameter must be a 2D or 3D tensor. For a 2D tensor, the shape is `(BS, H)`. For a 3D tensor, the product of the first two dimensions must be equal to BS, and the third dimension must be equal to H.
    <!-- end id17 -->

- **`group_tp`** (`str`): Optional. TP communication domain name, that is, the communication domain for data parallelism. This parameter is required only when TP domain communication is involved.

    <!-- npu="950,910b" id18 -->
    - <term>Atlas A2 products</term>, <term>Ascend 950DT products</term>: TP domain communication is not supported. Use the default value.
    <!-- end id18 -->
    <!-- npu="A3" id19 -->
    - <term>Atlas A3 products</term>: When TP domain communication is involved, the string length range is [1, 128), and the value must differ from `group_ep`.
    <!-- end id19 -->

- **`tp_world_size`** (`int`): Optional. Size of the TP communication domain. This parameter is required only when TP domain communication is involved.

    <!-- npu="910b" id20 -->
    - <term>Atlas A2 products</term>: TP domain communication is not supported. Use the default value `0`.
    <!-- end id20 -->
    <!-- npu="A3" id21 -->
    - <term>Atlas A3 products</term>: When TP domain communication is involved, the value range is [0, 2]. The values `0` and `1` indicate that there is no TP domain communication, and the value `2` indicates that there is TP domain communication.
    <!-- end id21 -->
    <!-- npu="950" id22 -->
    - <term>Ascend 950DT products</term>: Not supported in the current version. Pass 1.
    <!-- end id22 -->

- **`tp_rank_id`** (`int`): Optional. Rank ID of the current rank within the TP communication domain. This parameter is required only when TP domain communication is involved. If TP domain communication is not supported, use the default value `0`.

    <!-- npu="950,910b" id23 -->
    - <term>Atlas A2 products</term> and <term>Ascend 950DT products</term>: TP domain communication is not supported.
    <!-- end id23 -->
    <!-- npu="A3" id24 -->
    - <term>Atlas A3 products</term>: When TP domain communication is involved, the value range is [0, 1]. The `tp_rank_id` values of all ranks within the same TP communication domain must be unique. If there is no TP domain communication, pass `0`.
    <!-- end id24 -->

- **`expert_shard_type`** (`int`): Optional. Layout type of shared-expert ranks. The default value is 0.

    <!-- npu="910b" id25 -->
    - <term>Atlas A2 products</term>: This parameter is not supported currently. Use the default value.
    <!-- end id25 -->
    <!-- npu="950,A3" id26 -->
    - <term>Atlas A3 products</term> and <term>Ascend 950DT products</term>: Currently, only `0` is supported, indicating that shared-expert ranks are placed before MoE expert ranks.
    <!-- end id26 -->

- **`shared_expert_num`** (`int`): Optional. Number of shared experts, where a shared expert can be replicated and deployed across multiple ranks. The default value is 1.

    <!-- npu="910b" id27 -->
    - <term>Atlas A2 products</term>: This parameter is not supported currently. Use the default value.
    <!-- end id27 -->
    <!-- npu="A3" id28 -->
    - <term>Atlas A3 products</term>: The value range is [0, 4]. A value of `0` indicates that there is no shared expert.
    <!-- end id28 -->
    <!-- npu="950" id29 -->
    - <term>Ascend 950DT products</term>: In the current version, only 1 is supported.
    <!-- end id29 -->

- **`shared_expert_rank_num`** (`int`): Optional. Number of shared-expert ranks.

    <!-- npu="910b" id30 -->
    - <term>Atlas A2 products</term>: Shared experts are not supported. Set it to `0`.
    <!-- end id30 -->
    <!-- npu="950,A3" id31 -->
    - <term>Atlas A3 products</term> and <term>Ascend 950DT products</term>: The value range is [0, ep_world_size-1). A value of `0` indicates that there is no shared expert; a non-zero value requires `ep_world_size % shared_expert_rank_num = 0`.
    <!-- end id31 -->

- **`global_bs`** (`int`): Optional. Global batch size within the EP domain.

    <!-- npu="910b" id32 -->
    - <term>Atlas A2 products</term>: When the batch sizes differ across ranks, passing `max_bs * ep_world_size` or `256 * ep_world_size` is supported, where `max_bs` indicates the maximum batch size of a single rank. Passing `max_bs * ep_world_size` is recommended. Consistently passing `256 * ep_world_size` will result in a lack of support in future versions for scenarios where the batch size is greater than 256. When the batch sizes are identical across ranks, passing `0` or `BS * ep_world_size` is supported.
    <!-- end id32 -->
    <!-- npu="950,A3" id33 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: When the BS values of each rank are different, `max_bs*ep_world_size` can be passed, where `max_bs` indicates the maximum BS of a single rank. When the BS values of each rank are the same, the value can be 0 or `BS*ep_world_size`.
    <!-- end id33 -->

- **`out_dtype`** (`int`): Optional. **Reserved parameter, currently not used. Retain the default value.**
- **`comm_quant_mode`** (`int`): Optional. Communication quantization type.

    <!-- npu="910b" id34 -->
    - <term>Atlas A2 products</term>: The value can be `0` or `2`. `0` disables quantization during communication, and `2` enables `torch.int8` quantization during communication. The value `2` is supported only when `HCCL_INTRA_PCIE_ENABLE` is `1`, `HCCL_INTRA_ROCE_ENABLE` is `0`, and the driver version is 25.0.RC1.1 or later.
    <!-- end id34 -->
    <!-- npu="950,A3" id35 -->
    - <term>Atlas A3 products</term> and <term>Ascend 950DT products</term>: The value can be `0` or `2`. `0` disables quantization during communication, and `2` enables `torch.int8` quantization during communication. `torch.int8` quantization can be enabled only when `tp_world_size` is not `2`.
    <!-- end id35 -->

- **`group_list_type`** (`int`): Optional. **Reserved parameter, currently not used. Retain the default value.**

## Return Values

`Tensor`

Processed tokens. This parameter must be 2D with shape `(BS, H)`. The data type must be identical to that of the input `expand_x`. The data layout is ND. Non-contiguous tensors are not supported. The data type can be `torch.bfloat16` or `torch.float16`.

## Constraints

- This API can be used in inference scenarios.
<!-- npu="950" id36 -->
- Communication mode constraints:

    - <term>Ascend 950DT products</term>: Only UB Memory communication is supported.
<!-- end id36 -->

- This API supports single-operator mode and static graph mode, and `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` must be used in combination.
- Element values inside the `expand_idx`, `ep_recv_counts`, `tp_recv_counts`, and `expand_scales` tensors output by `npu_moe_distribute_dispatch` can vary across different product models, communication algorithms, or software versions. When using this API, pass these tensors directly to the corresponding parameters of `npu_moe_distribute_combine`. Other service logic of the model must not depend on them.
- The values of the `group_ep`, `ep_world_size`, `moe_expert_num`, `group_tp`, `tp_world_size`, `expert_shard_type`, `shared_expert_num`, `shared_expert_rank_num`, and `global_bs` parameters must remain identical across all ranks during the API call process. In addition, the values of `group_ep`, `ep_world_size`, `group_tp`, `tp_world_size`, `expert_shard_type`, and `global_bs` must also remain identical across different layers in the network and must match the corresponding parameters in [torch_npu.npu_moe_distribute_dispatch](torch_npu-npu_moe_distribute_dispatch.md).

<!-- npu="A3" id37 -->
- <term>Atlas A3 products</term>: In this scenario a single device contains two DIE (also called a "chiplet" or "bare die"), so "the current device" in the parameter descriptions refers to a single DIE.
<!-- end id37 -->
- The variables used in the shapes in the parameter descriptions are as follows:
    - A: Maximum number of tokens that can be received by the current rank. The value range is as follows:
        - For shared experts, `A=global_bs*shared_expert_num/shared_expert_rank_num` must be satisfied.
        - For MoE experts, when `global_bs` is `0`, `A>=BS*ep_world_size*min(local_expert_num, K)` must be satisfied; when `global_bs` is not `0`, `A>=global_bs*min(local_expert_num, K)` must be satisfied.

    - H: hidden size.

        <!-- npu="910b" id38 -->
        - <term>Atlas A2 products</term>: The value range is (0, 7168], and the value must be an integer multiple of 32.
        <!-- end id38 -->
        <!-- npu="A3" id39 -->
        - <term>Atlas A3 products</term>: Only 7168 is supported.
        <!-- end id39 -->
        <!-- npu="950" id40 -->
        - <term>Ascend 950DT products</term>: Only 7168 is supported.
        <!-- end id40 -->

    - BS: Number of tokens to be sent.

        <!-- npu="910b" id41 -->
        - <term>Atlas A2 products</term>: The value range is 0 < BS <= 256.
        <!-- end id41 -->
        <!-- npu="950,A3" id42 -->
        - <term>Atlas A3 products</term>, <term>Ascend 950DT products</term>: The value range is 0 < BS <= 512.
        <!-- end id42 -->

    - K: number of top-K experts selected, and 0 < K <= moe_expert_num must be satisfied.

        <!-- npu="910b" id43 -->
        - <term>Atlas A2 products</term>: The value range must be 0 < K <= 16.
        <!-- end id43 -->
        <!-- npu="A3" id44 -->
        - <term>Atlas A3 products</term>: The value range must be 0 < K <= 8.
        <!-- end id44 -->
        <!-- npu="950" id45 -->
        - <term>Ascend 950DT products</term>: The value range is 0 < K <= 8 and less than moe_expert_num.
        <!-- end id45 -->

    - server_num: number of server nodes. Only the values 2, 4, and 8 are supported.

        <!-- npu="910b" id46 -->
        - <term>Atlas A2 products</term>: This variable is used only in shapes for this scenario.
        <!-- end id46 -->

    - local_expert_num: Number of experts on the current rank.
        - For shared-expert ranks, `local_expert_num = 1`.
        - For MoE expert ranks, `local_expert_num = moe_expert_num/(ep_world_size - shared_expert_rank_num)`. When `local_expert_num > 1`, communication in the TP domain is not supported.

- HCCL communication domain buffer size:

    Before calling this API, verify that the configured HCCL_BUFFSIZE environment variable is reasonable.

    > [!NOTE]
    > CANN environment variable HCCL_BUFFSIZE: memory size occupied by a single communication domain, in MB. The default value is 200 MB if not configured.

    <!-- npu="910b" id47 -->
    - <term>Atlas A2 products</term>: The value must be >= 2*(BS*ep_world_size*min(local_expert_num, K)*H*sizeof(`torch.uint16`)+2MB).
    <!-- end id47 -->
    <!-- npu="A3" id48 -->
    - <term>Atlas A3 products</term>: The value must be \>=2 and must satisfy \>=2\*(local\_expert\_num\*max\_bs\*ep\_world\_size\*Align512(Align32(2\*H)+64)+(K+shared\_expert\_num)\*max\_bs\*Align512(2\*H)), where local\_expert\_num must be the number of local experts on the MoE expert device.
    <!-- end id48 -->
    <!-- npu="950" id49 -->
    - <term>Ascend 950DT products</term>: The value must be >=2 and must satisfy 1024^2*(HCCL_BUFFSIZE-2)/2 >= BS*2*(H+128)*(ep_world_size*local_expert_num+K+1), where local_expert_num must be the number of experts on the current MoE expert rank.
    <!-- end id49 -->

<!-- npu="910b" id50 -->
- <term>Atlas A2 products</term>: Configuring the environment variables `HCCL_INTRA_PCIE_ENABLE=1` and `HCCL_INTRA_ROCE_ENABLE=0` can reduce the cross-server communication data volume and improve operator performance. In this case, the condition `HCCL_BUFFSIZE >= moe_expert_num * bs * (H * sizeof(dtypeX) + 4 * ((K + 7) / 8 * 8) * sizeof(`torch.uint32`)) + 4MB + 100MB` must be satisfied. Furthermore, for the input parameter `moe_expert_num`, only the condition `moe_expert_num % (ep_world_size - shared_expert_rank_num) == 0` must be satisfied, and the condition `moe_expert_num / (ep_world_size - shared_expert_rank_num) <= 24` is not required.
<!-- end id50 -->
- In formulas in this document, the division sign (/) indicates integer division.
    - The `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` operators within a single model must operate in the same EP communication domain, which must not include other operators.
    - The `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` operators within a single model must either operate in the same TP communication domain or both operate without a TP communication domain. When a TP communication domain is involved, this domain must not include other operators.

    <!-- npu="A3" id51 -->
    - <term>Atlas A3 products</term>: Nodes in a communication domain must reside within the same SuperPoD. Cross-SuperPoD deployment is not supported.
    <!-- end id51 -->

<!-- npu="910b" id52 -->
- Networking constraints:

    - <term>Atlas A2 products</term>: In multi-server deployments, only switch-based networking is supported. Direct server-to-server connections are not supported.
<!-- end id52 -->

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
