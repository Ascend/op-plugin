# torch_npu.npu_moe_distribute_combine_add_rms_norm

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->

## Function<a name="en-us_topic_0000002322738573_section1470016430218"></a>

- **API function**: Fuses moe_distribute_combine, add, and rms_norm. It must be used together with [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md). It is equivalent to returning along the same path on which the npu_moe_distribute_dispatch_v2 operator collects data and then performing the add_rms_norm operation on the data. For the parameter pairing relationship with [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md), see [Parameter Description](#en-us_topic_0000002322738573_section187018431529) and [Constraints](#en-us_topic_0000002322738573_section470214314214).

> [!NOTE]
> If the pairing relationship is not followed, route recovery errors or communication result errors may occur.

- **Formulas**:
    - Data aggregation:

        $$
        rs\_out = ReduceScatterV(expand\_x)
        $$

        $$
        ata\_out = AlltoAllv(rs\_out)
        $$

        $$
        combine\_out = Sum(expert\_scales * ata\_out + expert\_scales * shared\_expert\_x)
        $$

        $$
        x = combine\_out + residual\_x
        $$

        $$
        y = \frac{x}{RMS(x)} * gamma, \quad \text{where } RMS(x) = \sqrt{\frac{1}{H}\sum_{i=1}^{H}x_i^2 + norm\_eps}
        $$

    - Special expert scenarios:
        - Zero expert scenario (`zero_expert_num ≠ 0`):

            $$
            Moe(ori\_x)=0
            $$

        - Copy-expert scenario (`copy_expert_num ≠ 0`):

            $$
            Moe(ori\_x)=ori\_x
            $$

        - Constant-expert scenario (`const_expert_num ≠ 0`):

            $$
            Moe(ori\_x)=const\_expert\_alpha\_1*ori\_x+const\_expert\_alpha\_2*const\_expert\_v
            $$

## Prototype<a name="en-us_topic_0000002322738573_section470115437220"></a>

```python
torch_npu.npu_moe_distribute_combine_add_rms_norm(expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, residual_x, gamma, group_ep, ep_world_size, ep_rank_id, moe_expert_num, *, tp_send_counts=None, x_active_mask=None, activation_scale=None, weight_scale=None, group_list=None, expand_scales=None, shared_expert_x=None, elastic_info=None, ori_x=None, const_expert_alpha_1=None, const_expert_alpha_2=None, const_expert_v=None, group_tp="", tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, global_bs=0, out_dtype=0, comm_quant_mode=0, group_list_type=0, comm_alg="", norm_eps=1e-06, zero_expert_num=0, copy_expert_num=0, const_expert_num=0) -> (Tensor, Tensor, Tensor)
```

## Parameters<a name="en-us_topic_0000002322738573_section187018431529"></a>

- **`expand_x`** (`Tensor`): Required. Token features expanded according to `expert_ids`. This parameter must be 2D with shape `(max(tp_world_size, 1) * A, H)`. The data type can be `torch.bfloat16`. The data layout can be ND. Non-contiguous tensors are supported.
- **`expert_ids`** (`Tensor`): Required. Top-K expert indices for each token. This parameter must be 2D with shape `(BS, K)`. The data type can be `torch.int32`. The data layout can be ND. Non-contiguous tensors are supported. This parameter corresponds to the `expert_ids` input of `torch_npu.npu_moe_distribute_dispatch`. The value range of elements inside the tensor is [0, `moe_expert_num`), and the $K$ values within the same row must be unique.
- **`expand_idx`** (`Tensor`): Required. Number of tokens dispatched to each expert. This parameter must be 1D with shape `(A * 128,)`. The data type can be `torch.int32`. The data layout can be ND. Non-contiguous tensors are supported. This parameter corresponds to the `assist_info_for_combine` output of [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md).
- **`ep_send_counts`** (`Tensor`): Required. Data amount sent from each expert on the current rank to each rank within the Expert Parallelism (EP) communication domain. This parameter must be a 1D tensor. The data type can be `torch.int32`. The data layout can be ND. Non-contiguous tensors are supported. This parameter corresponds to the `ep_recv_counts` output of `torch_npu.npu_moe_distribute_dispatch`. EP is a parallelism method specific to MoE models, in which different experts are distributed across different devices, with each device holding only a subset of the complete set of experts. Each token is routed to the experts that need to process it and sent to the devices hosting those experts through All-to-All communication for computation.
- **`expert_scales`** (`Tensor`): Required. Weights of the top-K experts for each token. This parameter must be 2D with shape `(BS, K)`. Shared-expert configurations do not require a weight factor and are summed directly. The data type can be `torch.float`. The data layout can be ND. Non-contiguous tensors are supported.
- **`residual_x`** (`Tensor`): Required. Residual parameter to be added to the processed tokens. This parameter must be 3D with shape `(BS, 1, H)`. The data type can be `torch.bfloat16`. The data layout can be ND. Non-contiguous tensors are supported.
- **`gamma`** (`Tensor`): Required. Weight parameter for `rms_norm`. This parameter must be 1D with shape `(H,)`. The data type can be `torch.bfloat16`. The data layout can be ND. Non-contiguous tensors are supported.
- **`group_ep`** (`str`): Required. EP communication domain name used for expert parallelism. The string length value range is [1, 128). The value of this parameter must differ from `group_tp`.
- **`ep_world_size`** (`int`): Required. Size of the EP communication domain. The value range is [2, 768].
- **`ep_rank_id`** (`int`): Required. Rank ID of the current rank within the EP communication domain. The value range is [0, `ep_world_size`). The `ep_rank_id` values of all ranks within the identical EP communication domain must be unique.
- **`moe_expert_num`** (`int`): Required. Number of MoE experts. The value range is [1, 1024], and the condition `moe_expert_num % (ep_world_size - shared_expert_rank_num) == 0` must be satisfied.
- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`tp_send_counts`** (`Tensor`): Optional. Amount of data sent from each expert on the current rank to each rank within the Tensor Parallelism (TP) communication domain. This is a reserved parameter that is not supported in the current version. Pass the default value `None`.
- **`x_active_mask`** (`Tensor`): Optional. Specifies whether a token participates in communication. By default, all tokens participate in communication. This parameter must be 1D or 2D. When the input is 1D, the shape is `(BS,)`; when the input is 2D, the shape is `(BS, K)`. The data type can be `torch.bool`. The data layout must be ND. Non-contiguous tensors are supported. When the input is 1D, the value `true` indicates that the corresponding token participates in communication, and all `true` values must precede any `false` values. For example, `{true, false, true}` is an invalid input. When the input is 2D, the value `true` indicates that the `expert_ids` entry corresponding to the current token participates in communication. If all K `torch.bool` values for a token are `false`, the token does not participate in communication. When the `BS` values differ across ranks, all tokens must be valid.
- **`activation_scale`** (`Tensor`): Optional. **Reserved parameter, currently not used. Retain the default value.**
- **`weight_scale`** (`Tensor`): Optional. **Reserved parameter, currently not used. Retain the default value.**
- **`group_list`** (`Tensor`): Optional. **Reserved parameter, currently not used. Retain the default value.**
- **`expand_scales`** (`Tensor`): Optional. This parameter corresponds to the `expand_scales` output of [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md). This parameter is not supported currently. Use the default value.
- **`shared_expert_x`** (`Tensor`): Optional. The data type must be identical to that of `expand_x`. This parameter is the shared-expert token data that must be added during the combine operation. Use it only when the number of shared-expert devices `shared_expert_rank_num` is `0`.
- **`elastic_info`** (`Tensor`): Optional. Reserved parameter, currently not used. Retain the default value `None`.
- **`ori_x`** (`Tensor`): Optional. Token data before FFN processing. This parameter is required when `copy_expert` or `const_expert` is enabled. You can choose to pass a valid tensor or a null pointer. When `copy_expert_num` or `const_expert_num` is not 0, a valid tensor must be provided. This parameter must be 2D with shape `(BS, H)`. The data type must be identical to that of `expand_x`. The data layout must be ND. Non-contiguous tensors are supported.
- **`const_expert_alpha_1`** (`Tensor`): Optional. Calculation coefficient required when `const_expert` is enabled. You can choose to pass a valid tensor or `None`. When `const_expert_num` is not 0, a valid tensor must be provided. This parameter must be 2D with shape `(const_expert_num, H)`. The data type must be identical to that of `expand_x`. The data layout must be ND. Non-contiguous tensors are supported.
- **`const_expert_alpha_2`** (`Tensor`): Optional. Calculation coefficient required when `const_expert` is enabled. You can choose to pass a valid tensor or `None`. When `const_expert_num` is not 0, a valid tensor must be provided. This parameter must be 2D with shape `(const_expert_num, H)`. The data type must be identical to that of `expand_x`. The data layout must be ND. Non-contiguous tensors are supported.
- **`const_expert_v`** (`Tensor`): Optional. Calculation coefficient required when `const_expert` is enabled. You can choose to pass a valid tensor or `None`. When `const_expert_num` is not 0, a valid tensor must be provided. This parameter must be 2D with shape `(const_expert_num, H)`. The data type must be identical to that of `expand_x`. The data layout must be ND. Non-contiguous tensors are supported.
- **`group_tp`** (`str`): Optional. Communication domain name used for tensor parallelism. This parameter is required only when TP domain communication is involved.
- **`tp_world_size`** (`int`): Optional. Size of the TP communication domain. This is a reserved parameter that is not supported in the current version. Pass the default value 0.
- **`tp_rank_id`** (`int`): Optional. Rank ID of the current rank within the TP communication domain. This is a reserved parameter that is not supported in the current version. Pass the default value 0.
- **`expert_shard_type`** (`int`): Optional. Layout type of shared-expert ranks. Currently, only `0` is supported, indicating that shared-expert ranks are placed before MoE expert ranks.
- **`shared_expert_num`** (`int`): Optional. Number of shared experts, where a shared expert can be replicated and deployed across multiple ranks. **Reserved parameter, currently not used. Only the default value 0 is supported.**
- **`shared_expert_rank_num`** (`int`): Optional. **Reserved parameter, currently not used. Only the default value `0` is supported.**
- **`global_bs`** (`int`): Optional. Global batch size within the EP domain. When the BS values of each rank are different, `max_bs*ep_world_size` can be passed, where `max_bs` indicates the maximum BS of a single rank. When the BS values of each rank are the same, the value can be 0 or `BS*ep_world_size`.
- **`out_dtype`** (`int`): Optional. **Reserved parameter, currently not used. Retain the default value.**
- **`comm_quant_mode`** (`int`): Optional. Communication quantization type. **Reserved parameter, currently not used. Retain the default value.**
- **`group_list_type`** (`int`): Optional. **Reserved parameter, currently not used. Retain the default value.**
- **`comm_alg`** (`str`): Optional. **Reserved parameter, currently not used. Retain the default value.**
- **`norm_eps`** (`float`): Optional. Epsilon value used to prevent division-by-zero errors in `add_rms_norm`. The default value is `1e-6`.
- **`zero_expert_num`** (`int`): Optional. Number of zero experts. The value range is [0, `MAX_INT32`), where the value of `MAX_INT32` is 2147483647. Valid zero-expert ID values are in the range [`moe_expert_num`, `moe_expert_num + zero_expert_num`).
- **`copy_expert_num`** (`int`): Optional. Number of copy experts. The value range is [0, `MAX_INT32`), where the value of `MAX_INT32` is `2147483647`. Valid copy-expert ID values are in the range [`moe_expert_num + zero_expert_num`, `moe_expert_num + zero_expert_num + copy_expert_num`).
- **`const_expert_num`** (`int`): Optional. Number of constant experts. The value range is [0, `MAX_INT32`), where the value of `MAX_INT32` is `2147483647`. Valid constant-expert ID values are in the range [`moe_expert_num` + `zero_expert_num + copy_expert_num, moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num`).

## Return Values<a name="en-us_topic_0000002322738573_section1370204314220"></a>

- **`y`** (`Tensor`): Result of applying `add_rms_norm` to the token after combine processing. This parameter must be 3D with shape `(BS, 1, H)`. The data type must be identical to that of the input `residual_x`. The data layout can be ND. Non-contiguous tensors are not supported.
- **`rstd_out`** (`Tensor`): Output of `add_rms_norm`. This parameter must be 3D with shape `(BS, 1, 1)`. The data type can be `torch.float`. The data layout can be ND. Non-contiguous tensors are not supported.
- **`x`** (`Tensor`): Result of the add operation performed on the token after combine processing. This parameter must be 3D with shape `(BS, 1, H)`. The data type must be identical to that of the input `residual_x`. The data layout can be ND. Non-contiguous tensors are not supported.

## Constraints<a name="en-us_topic_0000002322738573_section470214314214"></a>

- This API can be used in inference scenarios.
<!-- npu="950" id9 -->
- This API supports graph mode.
    - <term>Ascend 950DT products</term>: Only UB Memory communication is supported.
<!-- end id9 -->

- This API supports single-operator mode and static graph mode.
- The values of `expert_ids`, `x_active_mask`, `elastic_info`, `group_ep`, `ep_world_size`, `moe_expert_num`, `group_tp`, `tp_world_size`, `expert_shard_type`, `shared_expert_num`, `shared_expert_rank_num`, `global_bs`, `comm_quant_mode`, `zero_expert_num`, `copy_expert_num`, `const_expert_num`, and `HCCL_BUFFSIZE`, must be identical across all ranks during API execution. These values must also remain consistent across different layers of the network and match the corresponding parameters of [torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md).
- moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num < MAX_INT32, where the value of MAX_INT32 is 2147483647.

<!-- npu="A3" id10 -->
- <term>Atlas A3 products</term>: In this scenario, a single rank contains dual dies. Therefore, "this rank" in the parameter description indicates a single die.
<!-- end id10 -->
- Variables used in parameter tensor shapes:
    - `A`: Maximum number of tokens that can be received by the current rank. The value range is as follows:
        - When the value of `global_bs` is `0`, the condition `A >= bs * ep_world_size * min(local_expert_num, K)` must be satisfied.
        - When the value of `global_bs` is not `0`, the condition `A >= global_bs * min(local_expert_num, K)` must be satisfied.

    - Atlas A3 training products/Atlas A3 inference products: The value range is [1024, 8192].
    - Atlas A3 training products/Atlas A3 inference products: The value range is 0 < `BS` <= 512.
    - K: number of top-K experts selected. The value range is 0 < K <= 16, and 0 < K <= moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num must also be satisfied.
    - `K`: Number of top-K experts selected. The value range is 0 < `K` <= 8, and the condition 0 < `K` <= `moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num` must be satisfied.
        - For shared-expert ranks, `local_expert_num = 1`.
        - For MoE expert ranks, `local_expert_num = moe_expert_num / (ep_world_size - shared_expert_rank_num)`. When `local_expert_num > 1`, communication in the TP domain is not supported.

- HCCL communication domain buffer size:

    Before calling this API, check whether the value of the `HCCL_BUFFSIZE` environment variable is appropriate. This environment variable specifies the amount of memory occupied by a single communication domain, in MB, and defaults to 200 MB when not configured. In this scenario, the communication domain buffer size can be configured through the `HCCL_BUFFSIZE` environment variable or through [`hccl_buffer_size`](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/docs/en/performance_tuning/performance_tuning_methods/communication_basics_overview.md#hccl_buffer_size).

    - Within the EP communication domain: The value must be greater than or equal to 2 and be greater than or equal to `2 * (local_expert_num * max_bs * ep_world_size * Align512(Align32(2 * H) + 64) + (K + shared_expert_num) * max_bs * Align512(2 * H))`, where `local_expert_num` is the number of experts on a local MoE expert rank.
    - The alignment functions are defined as follows: $Align480(x) = ((x + 480 - 1) / 480) * 512$, $Align512(x) = ((x + 512 - 1) / 512) * 512$, $Align32(x) = ((x + 32 - 1) / 32) * 32$.

- Communication domain usage constraints:
    - The `npu_moe_distribute_dispatch_v2` and `npu_moe_distribute_combine_add_rms_norm` operators within a single model must operate in the same EP communication domain, which must not include other operators.
    <!-- npu="A3" id13 -->
    - <term>Atlas A3 products</term>: Nodes in a communication domain must reside within the same SuperPoD. Cross-SuperPoD deployment is not supported.
    <!-- end id13 -->

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
    sharedExpertRankNum = 0                      # Number of shared experts
    moeExpertNum = 32                            # Number of MoE experts
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

        bs_local = expert_ids.shape[0]
        torch.manual_seed(42)
        residual_x = torch.rand((bs_local, h), dtype=torch.bfloat16).npu()
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
            expert_scales=expert_scales,
            group_ep=ep_hcomm_info,
            ep_world_size=ep_world_size,
            ep_rank_id=rank // tp_world_size,
            expert_shard_type=0,
            shared_expert_num=0,
            shared_expert_rank_num=sharedExpertRankNum,
            moe_expert_num=moeExpertNum,
            global_bs=globalBS,
        )
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

    import time
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
    sharedExpertRankNum = 0                      # Number of shared experts
    moeExpertNum = 32                            # Number of MoE experts
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
                    scales, quant_mode, global_bs, expert_scales, residual_x, gamma, norm_eps):
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
            y, rstd_out, x = torch_npu.npu_moe_distribute_combine_add_rms_norm(expand_x=expand_x_npu,
                                                                               expert_ids=expert_ids,
                                                                               expand_idx=expand_idx_npu,
                                                                               ep_send_counts=ep_recv_counts_npu,
                                                                               expert_scales=expert_scales,
                                                                               group_ep=group_ep,
                                                                               ep_world_size=ep_world_size,
                                                                               ep_rank_id=ep_rank_id,
                                                                               expert_shard_type=expert_shard_type,
                                                                               shared_expert_rank_num=shared_expert_rank_num,
                                                                               moe_expert_num=moe_expert_num,
                                                                               global_bs=global_bs,
                                                                               shared_expert_num=0,
                                                                               residual_x=residual_x,
                                                                               gamma=gamma,
                                                                               norm_eps=norm_eps)

            return [y, rstd_out, x]

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

        bs_local = expert_ids.shape[0]
        torch.manual_seed(42)
        residual_x = torch.rand((bs_local, 1, h), dtype=torch.bfloat16).npu()
        torch.manual_seed(random_seed)
        gamma = torch.ones(h).to(input_dtype).npu()
        norm_eps = 1e-6

        model = MOE_DISTRIBUTE_GRAPH_Model()
        model = model.npu()
        npu_backend = torchair.get_npu_backend()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        output = model.forward(x, expert_ids, ep_hcomm_info, tp_hcomm_info, ep_world_size, tp_world_size,
                               rank // tp_world_size, rank % tp_world_size, 0, sharedExpertRankNum, moeExpertNum, scales,
                               quant_mode, globalBS, expert_scales, residual_x, gamma, norm_eps)
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
