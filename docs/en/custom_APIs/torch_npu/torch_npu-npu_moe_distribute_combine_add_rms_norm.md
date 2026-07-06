# torch\_npu.npu\_moe\_distribute\_combine\_add\_rms\_norm<a name="en-us_topic_0000002384325441"></a>

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>| √   |

## Function<a name="en-us_topic_0000002322738573_section1470016430218"></a>

- Description:

    This API must be used together with [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md). It returns data along the original data collection path of the `npu_moe_distribute_dispatch_v2` operator and performs an `add_rms_norm` operation.
    For parameter mappings between this API and [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md), see [Parameters](#en-us_topic_0000002322738573_section187018431529) and [Constraints](#en-us_topic_0000002322738573_section470214314214).
    > [!NOTE]
    > Failure to follow the required parameter mappings can lead to routing restoration failures or incorrect communication results.

     - This API supports data aggregation by fusing the functionalities of `moe_distribute_combine`, `add`, and `rms_norm`.
     - It also supports special expert scenarios.
- Formulas:
    - Data aggregation:

        $$
        rsOut = ReduceScatterV(expand\_x)\\
        ataOut = AllToAllV(rs\_out)\\
        combineOut = Sum(expert\_scales * ata\_out + expert\_scales * shared\_expert\_x)\\
        x = combine\_out + residual\_x\\
        y = \frac{x}{RMS(x)} * gamma,\quad\text{where}RMS(x) = \sqrt{\frac{1}{H}\sum_{i=1}^{H}x_{i}^{2}+norm\_eps}\\
        $$

    - Special expert scenarios:

        - Zero expert scenario (`zero_expert_num ≠ 0`):

            $$Moe(ori\_x)=0$$

        - Copy-expert scenario (`copy_expert_num ≠ 0`):

            $$Moe(ori\_x)=ori\_x$$

        - Constant-expert scenario (`const_expert_num ≠ 0`):

            $$Moe(ori\_x)=const\_expert\_alpha\_1*ori\_x+const\_expert\_alpha\_2*const\_expert\_v$$

## Prototype<a name="en-us_topic_0000002322738573_section470115437220"></a>

```python
torch_npu.npu_moe_distribute_combine_add_rms_norm(expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, residual_x, gamma, group_ep, ep_world_size, ep_rank_id, moe_expert_num, *, tp_send_counts=None, x_active_mask=None, activation_scale=None, weight_scale=None, group_list=None, expand_scales=None, shared_expert_x=None, elastic_info=None, ori_x=None, const_expert_alpha_1=None, const_expert_alpha_2=None, const_expert_v=None, group_tp="", tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, global_bs=0, out_dtype=0, comm_quant_mode=0, group_list_type=0, norm_eps=1e-06, int zero_expert_num=0, int copy_expert_num=0, int const_expert_num=0) -> (Tensor, Tensor, Tensor)
```

## Parameters<a name="en-us_topic_0000002322738573_section187018431529"></a>

- **`expand_x`** (`Tensor`): Required. Token features expanded according to `expert_ids`. This parameter must be 2D with shape `(max(tp_world_size, 1) * A, H)`. The data type can be `bfloat16`. The data layout is ND. Non-contiguous tensors are supported.
- **`expert_ids`** (`Tensor`): Required. Top-K expert indices for each token. This parameter must be 2D with shape `(BS, K)`. The data type can be `int32`. The data layout is ND. Non-contiguous tensors are supported. This parameter corresponds to the `expert_ids` input of `torch_npu.npu_moe_distribute_dispatch`. The value range of elements inside the tensor is [0, `moe_expert_num`), and the $K$ values within the identical row must be unique.
- **`expand_idx`** (`Tensor`): Required. Number of tokens dispatched to each expert. This parameter must be 1D with shape `(A * 128,)`. The data type can be `int32`. The data layout is ND. Non-contiguous tensors are supported. This parameter corresponds to the `expand_idx` output of `torch_npu.npu_moe_distribute_dispatch`.
- **`ep_send_counts`** (`Tensor`): Required. Data amount sent from each expert on the current rank to each rank within the Expert Parallelism (EP) communication domain. This parameter must be a 1D tensor. The data type can be `int32`. The data layout is ND. Non-contiguous tensors are supported. This parameter corresponds to the `ep_recv_counts` output of `torch_npu.npu_moe_distribute_dispatch`. EP is a parallelization strategy specific to Mixture-of-Experts (MoE) networks, where different experts are distributed across different ranks and each rank holds only a subset of the complete expert set. Each token is routed to specify which experts must process it, and is sent to the corresponding rank through AlltoAll communication to complete computation.
    Atlas A3 training products/Atlas A3 inference products: The shape must be `(ep_world_size * max(tp_world_size, 1) * local_expert_num,)`.

- **`expert_scales`** (`Tensor`): Required. Weights of the top-K experts for each token. This parameter must be 2D with shape `(BS, K)`. Shared-expert configurations do not require a weight factor and are summed directly. The data type can be `float`. The data layout is ND. Non-contiguous tensors are supported.
- **`residual_x`** (`Tensor`): Required. Residual parameter to be added to the processed tokens. This parameter must be 3D with shape `(BS, 1, H)`. The data type can be `bfloat16`. The data layout is ND. Non-contiguous tensors are supported.
- **`gamma`** (`Tensor`): Required. Weight parameter for `rms_norm`. This parameter must be 1D with shape `(H,)`. The data type can be `bfloat16`. The data layout is ND. Non-contiguous tensors are supported.
- **`group_ep`** (`str`): Required. EP communication domain name used for expert parallelism. The string length value range is [1, 128). The value of this parameter must differ from `group_tp`.
- **`ep_world_size`** (`int`): Required. Size of the EP communication domain.
    - Atlas A3 training products/Atlas A3 inference products: The value range is [2, 768].

- **`ep_rank_id`** (`int`): Required. Rank ID of the current rank within the EP communication domain. The value range is [0, `ep_world_size`). The `ep_rank_id` values of all ranks within the identical EP communication domain must be unique.
- **`moe_expert_num`** (`int`): Required. Number of MoE experts. The value range is [1, 1024], and the condition `moe_expert_num % (ep_world_size - shared_expert_rank_num) == 0` must be satisfied.
- **`tp_send_counts`** (`Tensor`): Optional. Amount of data sent from each expert on the current rank to each rank within the Tensor Parallelism (TP) communication domain. This parameter corresponds to the `tp_recv_counts` output of `torch_npu.npu_moe_distribute_dispatch`. TP is a general model parallelism strategy in which the weights or activations of a single operator are partitioned across multiple ranks along a given dimension. The ranks collaboratively perform the computation, and the results are aggregated through collective communication operations such as AllReduce and AllGather.
    - Atlas A3 training products/Atlas A3 inference products: The TP communication domain is supported. This parameter must be 1D with shape `(tp_world_size,)`. The data type can be `int32`. The data layout must be ND. Non-contiguous tensors are supported.

- **`x_active_mask`** (`Tensor`):
    - Atlas A3 training products/Atlas A3 inference products: This parameter must be 1D or 2D. When the input is 1D, the shape of this parameter is `(BS,)`. When the input is 2D, the shape of this parameter is `(BS, K)`. The data type can be `bool`. The data layout must be ND. Non-contiguous tensors are supported. When the input is 1D, the value `True` indicates that the corresponding token participates in communication, and all `True` values must precede any `False` values. For example, `{True, False, True}` is an invalid input. When the input is 2D, the value `True` indicates that the `expert_ids` entry corresponding to the current token participates in communication. If all $K$ Boolean values for a token are `False`, the token does not participate in communication. By default, all tokens participate in communication. When the `BS` values differ across ranks, all tokens must be valid.

- **`activation_scale`** (`Tensor`): Optional. **Reserved parameter, currently not used. Retain the default value.**
- **`weight_scale`** (`Tensor`): Optional. **Reserved parameter, currently not used. Retain the default value.**
- **`group_list`** (`Tensor`): Optional. **Reserved parameter, currently not used. Retain the default value.**
- **`expand_scales`** (`Tensor`): Optional. This parameter corresponds to the `expand_scales` output of `torch_npu.npu_moe_distribute_dispatch`.
    - Atlas A3 training products/Atlas A3 inference products: Currently, this parameter is not supported. Retain the default value.

- **`shared_expert_x`** (`Tensor`): Optional. The data type must be identical to that of `expand_x`. This parameter is the shared-expert token data that must be added during the combine operation. Use it only when the number of shared-expert devices `shared_expert_rank_num` is `0`.
    - Atlas A3 training products/Atlas A3 inference products: The data type must match that of `expand_x`, and the shape of this parameter is `(BS, H)`.

- **`elastic_info`** (`Tensor`): Optional. Reserved parameter, currently not used. Retain the default value `None`.

- **`ori_x`** (`Tensor`): Optional. Token data before FFN processing. This parameter is required when `copy_expert` or `const_expert` is enabled. You can choose to pass a valid tensor or a null pointer. When `copy_expert_num` or `const_expert_num` is not 0, a valid tensor must be provided. This parameter must be 2D with shape `(BS, H)`. The data type must be identical to that of `expand_x`. The data layout must be ND. Non-contiguous tensors are supported.

- **`const_expert_alpha_1`** (`Tensor`): Optional. Calculation coefficient required when `const_expert` is enabled. You can choose to pass a valid tensor or `None`. When `const_expert_num` is not 0, a valid tensor must be provided. This parameter must be 2D with shape `(const_expert_num, H)`. The data type must be identical to that of `expand_x`. The data layout must be ND. Non-contiguous tensors are supported.

- **`const_expert_alpha_2`** (`Tensor`): Optional. Calculation coefficient required when `const_expert` is enabled. You can choose to pass a valid tensor or `None`. When `const_expert_num` is not 0, a valid tensor must be provided. This parameter must be 2D with shape `(const_expert_num, H)`. The data type must be identical to that of `expand_x`. The data layout must be ND. Non-contiguous tensors are supported.

- **`const_expert_v`** (`Tensor`): Optional. Calculation coefficient required when `const_expert` is enabled. You can choose to pass a valid tensor or `None`. When `const_expert_num` is not 0, a valid tensor must be provided. This parameter must be 2D with shape `(const_expert_num, H)`. The data type must be identical to that of `expand_x`. The data layout must be ND. Non-contiguous tensors are supported.

- **`group_tp`** (`str`): Optional. Communication domain name used for tensor parallelism. This parameter is required only when TP domain communication is involved.
    - Atlas A3 training products/Atlas A3 inference products: When there is TP domain communication, the string length value range is [0, 128). The value of this parameter must differ from `group_ep`. An empty value is supported only when there is no TP domain.

- **`tp_world_size`** (`int`): Optional. Size of the TP communication domain. This parameter is required only when TP domain communication is involved.
    - Atlas A3 training products/Atlas A3 inference products: When TP domain communication is involved, the value range is [0, 2]. The values `0` and `1` indicate that there is no TP domain communication, and the value `2` indicates that there is TP domain communication.

- **`tp_rank_id`** (`int`): Optional. Rank ID of the current rank within the TP communication domain. This parameter is required only when TP domain communication is involved.
    - Atlas A3 training products/Atlas A3 inference products: When TP domain communication is involved, the value range is [0, 1]. The `tp_rank_id` of each rank in the same TP communication domain must be unique. If there is no TP domain communication, pass `0`.

- **`expert_shard_type`** (`int`): Optional. Layout type of shared-expert ranks.
    - Atlas A3 training products/Atlas A3 inference products: Currently, only `0` is supported, indicating that shared-expert ranks are arranged in front of MoE expert ranks.

- **`shared_expert_num`** (`int`): Optional. Number of shared experts, where a shared expert can be replicated and deployed across multiple ranks. **Reserved parameter, currently not used. Only the default value `0` is supported.**
- **`shared_expert_rank_num`** (`int`): Optional. **Reserved parameter, currently not used. Only the default value `0` is supported.**
- **`global_bs`** (`int`): Optional. Global batch size within the EP communication domain.
    - Atlas A3 training products/Atlas A3 inference products: When the batch sizes differ across ranks, passing `max_bs * ep_world_size` is supported, where `max_bs` indicates the maximum batch size of a single rank. When the batch sizes are identical across ranks, passing `0` or `BS * ep_world_size` is supported.

- **`out_dtype`** (`int`): Optional. **Reserved parameter, currently not used. Retain the default value.**
- **`comm_quant_mode`** (`int`): Optional. Communication quantization type. **Reserved parameter, currently not used. Retain the default value.**
- **`group_list_type`** (`int`): Optional. **Reserved parameter, currently not used. Retain the default value.**
- **`norm_eps`** (`float`): Optional. Epsilon value used to prevent division-by-zero errors in `add_rms_norm`. The default value is `1e-6`.

- **`zero_expert_num`** (`int`): Optional. Number of zero experts. The value range is [0, `MAX_INT32`), where the value of `MAX_INT32` is 2147483647. Valid zero-expert ID values are in the range [`moe_expert_num`, `moe_expert_num + zero_expert_num`).

- **`copy_expert_num`** (`int`): Optional. Number of copy experts. The value range is [0, `MAX_INT32`), where the value of `MAX_INT32` is `2147483647`. Valid copy-expert ID values are in the range [`moe_expert_num + zero_expert_num`, `moe_expert_num + zero_expert_num + copy_expert_num`).

- **`const_expert_num`** (`int`): Optional. Number of constant experts. The value range is [0, `MAX_INT32`), where the value of `MAX_INT32` is `2147483647`. Valid constant-expert ID values are in the range [`moe_expert_num` + `zero_expert_num + copy_expert_num, moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num`).

## Return Values<a name="en-us_topic_0000002322738573_section1370204314220"></a>

- **`y`** (`Tensor`): Result of applying `add_rms_norm` to the token after combine processing. This parameter must be 3D with shape `(BS, 1, H)`. The data type must be identical to that of the input `residual_x`. The data layout is ND. Non-contiguous tensors are not supported.
- **`rstd_out`** (`Tensor`): Output of `add_rms_norm`. This parameter must be 3D with shape `(BS, 1, 1)`. The data type can be `float`. The data layout is ND. Non-contiguous tensors are not supported.
- **`x`** (`Tensor`): Result of the add operation performed on the token after combine processing. This parameter must be 3D with shape `(BS, 1, H)`. The data type must be identical to that of the input `residual_x`. The data layout is ND. Non-contiguous tensors are not supported.

## Constraints<a name="en-us_topic_0000002322738573_section470214314214"></a>

- This API can be used in inference scenarios.
- This API supports graph mode.
- The values of `expert_ids`, `x_active_mask`, `elastic_info`, `group_ep`, `ep_world_size`, `moe_expert_num`, `group_tp`, `tp_world_size`, `expert_shard_type`, `shared_expert_num`, `shared_expert_rank_num`, `global_bs`, `comm_alg`, `zero_expert_num`, `copy_expert_num`, `const_expert_num`, and `HCCL_BUFFSIZE` must be identical across all ranks during API execution. These values must also remain consistent across different layers of the network and match the corresponding parameters of [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md).
- Atlas A3 training products/Atlas A3 inference products: In this scenario, a single rank contains dual dies. Therefore, "this rank" in the parameter description indicates a single die.
- The condition `moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num < MAX_INT32` must be satisfied, where the value of `MAX_INT32` is `2147483647`.
- Variables used in parameter tensor shapes:
    - `A`: Maximum number of tokens that can be received by the current rank. The value range is as follows:
        - When the value of `global_bs` is `0`, the condition `A >= BS * ep_world_size * min(local_expert_num, K)` must be satisfied.
        - When the value of `global_bs` is not `0`, the condition `A >= global_bs * min(local_expert_num, K)` must be satisfied.

    - `H`: Hidden layer size.
        - Atlas A3 training products/Atlas A3 inference products: The value range is [1024, 8192].

    - `BS`: Number of tokens to be sent.
        - Atlas A3 training products/Atlas A3 inference products: The value range is 0 < `BS` <= 512.

    - `K`: Number of top-K experts selected. The value range is 0 < `K` <= 8, and the condition 0 < `K` <= `moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num` must be satisfied.
    - `local_expert_num`: Number of experts on the current rank.
        - For shared-expert ranks, `local_expert_num = 1`.
        - For MoE expert ranks, `local_expert_num = moe_expert_num / (ep_world_size - shared_expert_rank_num)`. When `local_expert_num > 1`, communication in the TP domain is not supported.

- HCCL communication domain buffer size:

    Before calling this API, ensure that the `HCCL_BUFFSIZE` environment variable is configured appropriately. This variable specifies the memory size occupied by a single communication domain, in MB. If not specified, the default value `200` MB is used. The buffer size can be configured through either the `HCCL_BUFFSIZE` environment variable or the `hccl_buffer_size` parameter. For details, see section "hccl_buffer_size" in [PyTorch Training Model Porting and Tuning](https://hiascend.com/document/redirect/canncommercial-ptmigr) (path: **Performance Profiling** > **Performance Profiling Methods** > **Communication Optimization** > **Optimization Methods** > **hccl_buffer_size**).
    - Within the EP communication domain: The value must be greater than or equal to 2 and be greater than or equal to `2 * (local_expert_num * max_bs * ep_world_size * Align512(Align32(2 * H) + 64) + (K + shared_expert_num) * max_bs * Align512(2 * H))`, where `local_expert_num` is the number of experts on a local MoE expert rank.
    - Within the TP communication domain: The value must be grater than or equal to `(A * Align512(Align32(h * 2) + 44) + A * Align512(h * 2)) * 2`.
    - The alignment functions are defined as follows: $Align480(x) = ((x + 480 - 1) / 480) * 512$, $Align512(x) = ((x + 512 - 1) / 512) * 512$, $Align32(x) = ((x + 32 - 1) / 32) * 32$.

- Communication domain usage constraints:

    - The `npu_moe_distribute_dispatch_v2` and `npu_moe_distribute_combine_add_rms_norm` operators within a single model must operate in the same EP communication domain, which must not include other operators.

    - The `npu_moe_distribute_dispatch_v2` and `npu_moe_distribute_combine_add_rms_norm` operators within a single model must either operate in the same TP communication domain or both operate without a TP communication domain. When a TP communication domain is involved, this domain must not include other operators.

    - Atlas A3 training products/Atlas A3 inference products: Nodes in a communication domain must reside within the same SuperPoD. Cross-SuperPoD deployment is not supported.

## Examples<a name="en-us_topic_0000002322738573_section9702174311218"></a>

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
    import time

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
    shared_expert_rank_num = 0                      # Number of shared experts
    moe_expert_num = 32                            # Number of MoE experts
    bs = 8                                       # Number of tokens
    h = 7168                                     # Length of each token
    k = 8
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - shared_expert_rank_num
    local_moe_expert_num = moe_expert_num // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)
    zero_expert_num = 0
    copy_expert_num = 0
    const_expert_num = 0

    def gen_const_expert_alpha_1():
        const_expert_alpha_1 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1

    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_2

    def gen_const_expert_v():
        const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_v

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
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        ep_group, tp_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        tp_hcomm_info = get_hcomm_info(rank, tp_group)

        # Create input tensors
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = torch.tensor([[ 5,  7, 17,  4,  2,  6, 11, 16],
            [10, 12, 13, 15, 19,  4, 18,  1],
            [19, 30,  1, 17,  9,  5,  0, 31],
            [19, 11, 17,  0, 10,  5,  7,  9],
            [10, 16, 11, 17, 30,  8,  9,  3],
            [12, 19,  5,  7,  1,  3, 18, 16],
            [11,  9, 13, 16, 12, 30, 17, 14],
            [16,  4,  9,  5,  0, 10, 11, 17]], dtype=torch.int32).npu()

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

        expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales = torch_npu.npu_moe_distribute_dispatch_v2(
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

        bs_local = expert_ids.shape[0]
        torch.manual_seed(42)
        residual_x = torch.rand((bs_local, 1, h), dtype=torch.bfloat16).npu()
        torch.manual_seed(random_seed)
        gamma = torch.ones(h).to(input_dtype).npu()
        norm_eps = 1e-6
        y, rstd_out, x = torch_npu.npu_moe_distribute_combine_add_rms_norm(
            expand_x=expand_x,
            residual_x=residual_x,
            gamma=gamma,
            norm_eps=norm_eps,
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
            shared_expert_num=0,
            shared_expert_rank_num=shared_expert_rank_num,
            moe_expert_num=moe_expert_num,
            global_bs=globalBS,
            ori_x=x,
            const_expert_alpha_1=const_expert_alpha_1,
            const_expert_alpha_2=const_expert_alpha_2,
            const_expert_v=const_expert_v,
            zero_expert_num=zero_expert_num,
            copy_expert_num=copy_expert_num,
            const_expert_num=const_expert_num
        )
        time.sleep(10)
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
    shared_expert_rank_num = 0                      # Number of shared experts
    moe_expert_num = 32                            # Number of MoE experts
    bs = 8                                       # Number of tokens
    h = 7168                                     # Length of each token
    k = 8
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - shared_expert_rank_num
    local_moe_expert_num = moe_expert_num // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)
    zero_expert_num = 0
    copy_expert_num = 0
    const_expert_num = 0

    class MOE_DISTRIBUTE_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, expert_ids, group_ep, group_tp, ep_world_size, tp_world_size,
                    ep_rank_id, tp_rank_id, expert_shard_type, shared_expert_rank_num, moe_expert_num,
                    scales, quant_mode, global_bs, expert_scales, residual_x, gamma, norm_eps, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v, zero_expert_num, copy_expert_num, const_expert_num):
            output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(x=x,
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
                                                                        const_expert_num=const_expert_num)

            expand_x_npu, _, expand_idx_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
            if expand_x_npu.dtype == torch.int8:
                expand_x_npu = expand_x_npu.to(input_dtype)
            y, rstd_out, x = torch_npu.npu_moe_distribute_combine_add_rms_norm(expand_x=expand_x_npu,
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
                                                                               global_bs=global_bs,
                                                                               shared_expert_num=0,
                                                                               residual_x=residual_x,
                                                                               gamma=gamma,
                                                                               norm_eps=norm_eps,
                                                                               elastic_info=elastic_info,
                                                                               ori_x=x,
                                                                               const_expert_alpha_1=const_expert_alpha_1,
                                                                               const_expert_alpha_2=const_expert_alpha_2,
                                                                               const_expert_v=const_expert_v,
                                                                               zero_expert_num=zero_expert_num,
                                                                               copy_expert_num=copy_expert_num,
                                                                               const_expert_num=const_expert_num)

            return [y, rstd_out, x]

    def gen_const_expert_alpha_1():
        const_expert_alpha_1 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1

    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_2

    def gen_const_expert_v():
        const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_v

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
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        ep_group, tp_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        tp_hcomm_info = get_hcomm_info(rank, tp_group)

        # Create input tensors
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = torch.tensor([[0, 8, 4, 1, 6, 12, 14, 17],
            [14, 10, 7, 3, 0, 12, 11, 17],
            [12, 0, 5, 11, 19, 4, 6, 18],
            [17, 3, 4, 10, 18, 0, 1, 2],
            [13, 16, 9, 10, 15, 6, 7, 14],
            [17, 15, 14, 8, 16, 18, 3, 12],
            [4, 12, 2, 17, 15, 3, 9, 10],
            [16, 7, 12, 9, 18, 3, 19, 17]], dtype=torch.int32).npu()

        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + moe_expert_num, h) if shared_expert_rank_num else (moe_expert_num, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None

        bs_local = expert_ids.shape[0]
        torch.manual_seed(42)
        residual_x = torch.rand((bs_local, 1, h), dtype=torch.bfloat16).npu()
        torch.manual_seed(random_seed)
        gamma = torch.ones(h).to(input_dtype).npu()
        norm_eps = 1e-6

        elastic_info = None
        const_expert_alpha_1 = gen_const_expert_alpha_1().npu()
        const_expert_alpha_2 = gen_const_expert_alpha_2().npu()
        const_expert_v = gen_const_expert_v().npu()

        out = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)
        model = MOE_DISTRIBUTE_GRAPH_Model()
        model = model.npu()
        npu_backend = torchair.get_npu_backend()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        output = model.forward(x, expert_ids, ep_hcomm_info, tp_hcomm_info, ep_world_size, tp_world_size,
                            rank // tp_world_size,rank % tp_world_size, 0, shared_expert_rank_num, moe_expert_num, scales,
                            quant_mode, globalBS, expert_scales, residual_x, gamma, norm_eps, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v, zero_expert_num, copy_expert_num, const_expert_num)
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
