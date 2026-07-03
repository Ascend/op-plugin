# torch_npu.npu_moe_update_expert

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |

## Function

Provides load balancing and expert pruning features, where the mapped expert table and mask can be passed into the MoE layer for data dispatching and processing.

- Load balancing: Maps the logical rank IDs of the top-K experts assigned to each token to physical rank IDs in redundant expert deployment scenarios. The computation method is as follows:

   The variables used in the code are defined as follows. `F` indicates the number of columns in `eplb_table`, and `ceil(a, b)` indicates ceiling division, $\lceil a/b \rceil$.

   For load balancing, the following computation is performed on the i-th value in `expert_ids` (the i-th token).

    ```python
    new_expert_id = eplb_table[table_offset + 1]
    expert_id = expert_ids[i]
    table_offset = expert_id * F
    place_num = eplb_table[table_offset]
    if (eplb_table[table_offset] == 1):
        new_expert_id = eplb_table[table_offset + 1]
    else:
        if (balance_mode == 0):
            mode_value = ceil(world_size, place_num)
            place_idx = local_rank_id / mode_value + 1
        else:
            place_idx = i % place_num
    new_expert_id = eplb_table[table_offset + place_idx]
    ```

- Expert pruning: The top K experts to which tokens are sent can be pruned based on the threshold. The computation method is as follows:
   
   The `active_mask` with shape `(BS,)` is broadcast to `active_mask_tensor` with shape `(BS, K)`, where experts corresponding to `False` are directly pruned. Elements of `expert_scales` where `active_mask_tensor` is `True` are also pruned when the condition is satisfied.

    ```python
    active_mask_tensor = broadcast(active_mask, (BS, K))
    for i in range(BS):
        expert_scales_vec[:] = sum(expert_scales[i, :]) * pruning_threshold[:]
        balanced_active_mask[i, :] = (expert_scales_vec[i, :] < expert_scales[:]) & active_mask_tensor[i, :]
    ```

## Prototype

```python
torch_npu.npu_moe_update_expert(expert_ids, eplb_table, *, expert_scales=None, pruning_threshold=None, active_mask=None, local_rank_id=-1, world_size=-1, balance_mode=0) -> (Tensor, Tensor)
```

## Parameters

- **`expert_ids`** (`Tensor`): Required. Top-K expert indices for each token. This parameter must be 2D with shape `(BS, K)`. The data type can be `int32` or `int64`. The data layout must be ND. Non-contiguous tensors are supported.
- **`eplb_table`** (`Tensor`): Required. Mapping table from logical experts to physical experts. The caller must ensure that the values in the input tensor are valid. The first column of each row stores the number of deployed instances (denoted by `count`) for the corresponding logical expert, and the value must be greater than or equal to `1`. Columns `[1, count]` store the physical rank IDs of the corresponding instances, with values in the range `[0, moe_expert_num)`. The shape of this parameter is `(log_expert_num, F)`. The data type can be `int32`. The data layout must be ND. Non-contiguous tensors are supported.  
  - **`log_expert_num`**: Number of logical experts, which is equal to the number of rows in `eplb_table`. Each logical expert corresponds to one row in the mapping table. The value range is (0, 1024).
  - **`F`**: Number of columns in the input mapping table. The value range is [2, `world_size + 1`]. The first column stores the number of deployed instances for the logical expert corresponding to each row (value > 0), and the remaining `F - 1` columns store the physical rank IDs on which the logical expert is deployed.
  - **`moe_expert_num`**: Total number of physical experts, indicating the total number of replicas deployed for all logical experts. That is, the sum of all `count` values in the first column of `eplb_table`. The value range is (0, 1024]. This parameter plays the same role as `moe_expert_num` in the `torch_npu.npu_moe_distribute_dispatch` and `torch_npu.npu_moe_distribute_dispatch_v2` APIs.
- **`expert_scales`** (`Tensor`): Optional. Scale weights of the top-K experts for each token. Ensure that the scale weights are sorted in descending order within each token. You can provide a valid tensor or a null pointer. When a valid tensor is provided for this parameter, you must also provide a valid tensor for `pruning_threshold`. The shape of this parameter is `(BS, K)`. The data type can be `fp16`, `bf16`, or `float. The data layout must be ND. Non-contiguous tensors are supported.
- **`pruning_threshold`** (`Tensor`): Optional. Minimum threshold for expert scale weights. If the scale weight of a top-K expert for a token is smaller than the corresponding threshold, that expert is pruned for the token. That is, the token is not dispatched to that expert for processing. A valid tensor or a null pointer can be provided. When a valid tensor is provided for this parameter, a valid tensor must also be provided for `expert_scales`. The shape of this parameter is `(K,)` or `(1, K)`. The data type can be `float`. The data layout must be ND. Non-contiguous tensors are supported.
- **`active_mask`** (`Tensor`): Optional. Indicates whether a token participates in communication. A valid tensor or a null pointer can be provided. If a valid tensor is provided, a valid tensor must also be provided for `expert_scales` and `pruning_threshold`. The value `true` indicates that the corresponding token participates in communication. All `true` values must appear before any `false` values. For example, `{true, false, true}` is an invalid input. If a null pointer is provided, all tokens participate in communication. The shape of this parameter is `(BS,)`. The data type can be `bool`. The data layout must be ND. Non-contiguous tensors are supported.

- **`local_rank_id`** (`int`): ID of the current rank. The data type is `int64`. When `balance_mode` is set to `0`, the value range is [0, `world_size`).
- **`world_size`** (`int`): Size of the communication domain. The data type can be `int64`. When `balance_mode` is set to `0`, the value range of this parameter is [2, 768].
- **`balance_mode`** (`int`): Load-balancing mode. The data type can be `int64`. Valid values are `0` (load balancing based on `local_rank_id`) or `1` (load balancing based on token_id). When this parameter is set to `0`, a valid tensor must be provided for both `local_rank_id` and `world_size`.

## Return Values

- **`balanced_expert_ids`** (`Tensor`): Physical rank IDs of the top-K experts for each token after mapping. The shape of this parameter is `(BS, K)`. The data type and data layout must be identical to those of `expert_ids`.
- **`balanced_active_mask`** (`Tensor`): Pruned `active_mask`. This output is valid only when valid tensors are provided for both `expert_scales` and `pruning_threshold`. The shape of this parameter is `(BS, K)`. The data type can be `bool`. The data layout must be ND. Non-contiguous tensors are supported.

## Constraints

- This API must be used together with the `torch_npu.npu_moe_distribute_dispatch` or `torch_npu.npu_moe_distribute_dispatch_v2` API.
- Values of the `world_size` and `moe_expert_num` parameters must remain identical across all ranks during the API call process. In addition, these values must also remain identical across different layers in the network. The following table describes the mapping between parameters in this API and those in `torch_npu.npu_moe_distribute_dispatch` or `torch_npu.npu_moe_distribute_dispatch_v2`.

    |`torch_npu.npu_moe_update_expert`     |`torch_npu.npu_moe_distribute_dispatch`/`torch_npu.npu_moe_distribute_dispatch_v2`|
    |---------------------------|-----------------------------------------------------------|
    |`local_rank_id`            |`ep_rank_id`                                               |
    |`world_size`               |`ep_world_size`                                            |
    |Sum of the count values in the first column of `eplb_table`|`moe_expert_num`                                           |
    |`BS`                       |`BS`                                                       |
    |`K`                        |`K`                                                        |

- <term>Atlas A3 training products/Atlas A3 inference products</term>: In this scenario, a single rank contains dual dies. Therefore, "this rank" in the parameter description indicates a single die.
- The shape-related variables used in the parameter descriptions are defined as follows:
    - **`BS`**: Batch sequence size, indicating the token count ultimately output by the current rank. For <term>Atlas A3 training products/Atlas A3 inference products</term>, the value range is `0 < BS <= 512`.
    - **`K`**: Top-K expert selection count. The conditions `0 < K <= 16` and `0 < K <= log_expert_num` must be satisfied.
    - **`log_expert_num`**: Number of logical experts, representing the number of rows in `eplb_table`. The value range is (0, 1024).
    - **`moe_expert_num`**: Total number of physical experts, indicating the total number of replicas deployed for all logical experts. That is, the sum of all `count` values in the first column of `eplb_table`. The value range is (0, 1024].
    - **`F`**: Number of columns in the input mapping table `eplb_table`. The value range is [2, `world_size + 1`].
    - Number of replicas deployed for each logical expert (the count value in the first column of `eplb_table`). The minimum value is `1` and the maximum value is `world_size`.
    - Total number of replicas deployed across all logical experts (sum of the count values in the first column of `eplb_table`). The total must be less than or equal to `1024` and be divisible by `world_size`.

## Examples

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
    dtype = torch.int32
    local_moe_expert_num = 4
    is_pruning = 1
    balance_mode = 1
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
    shared_expert_rank_num = 2                      # Number of shared experts
    BS = 8                                       # Number of tokens
    h = 7168                                     # Length of each token
    K = 8
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - shared_expert_rank_num
    moe_expert_num = moe_rank_num * local_moe_expert_num
    globalBS = BS * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)
    phy_ep_size = (ep_world_size - shared_expert_rank_num) * local_moe_expert_num
    log_ep_size = phy_ep_size // local_moe_expert_num
    F = random.randint(2, world_size + 1)
    
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
        x = torch.randn(BS, h, dtype=input_dtype).npu()
        eplb_table = np.zeros((log_ep_size, F - 1))
        count_column = np.random.randint(1, F, size=(log_ep_size, 1))
        all_ranks = np.arange(phy_ep_size)
        for i in range(log_ep_size):
            np.random.shuffle(all_ranks)
            for j in range(count_column[i][0]):
                eplb_table[i][j] = all_ranks[j]
        eplb_table = np.hstack((count_column, eplb_table))
        expert_ids = np.random.randint(low=0, high=log_ep_size, size=(BS, K))
        expert_scales = torch.randn(BS, K, dtype=torch.float32).npu()
        pruning_threshold = None
        active_mask = None
        if is_pruning:
            expert_scales_temp = -np.sort(-np.random.uniform(low=0, high=0.25, size=(BS, K)), axis=1)
            pruning_threshold_temp = np.random.uniform(low=0, high=0.15, size=(1, K))
            num_true = np.random.randint(0, BS + 1)
            active_mask_temp = np.concatenate([np.ones(num_true, dtype=bool), np.zeros(BS - num_true, dtype=bool)])
            expert_scales = torch.from_numpy(expert_scales_temp).to(torch.float32).npu()
            pruning_threshold = torch.from_numpy(pruning_threshold_temp).to(torch.float32).npu()
            active_mask = torch.from_numpy(active_mask_temp).to(torch.bool).npu()
        scales_shape = (1 + moe_expert_num, h) if shared_expert_rank_num else (moe_expert_num, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None
        
        eplb_table_tensor = torch.from_numpy(eplb_table).to(torch.int32)
        expert_ids_tensor = torch.from_numpy(expert_ids).to(dtype)

        npu_balanced_expert_ids, npu_balanced_active_mask = torch_npu.npu_moe_update_expert(
            expert_ids=expert_ids_tensor.npu(),
            eplb_table=eplb_table_tensor.npu(),
            expert_scales=expert_scales if is_pruning else None,
            pruning_threshold=pruning_threshold if is_pruning else None,
            active_mask=active_mask if is_pruning else None,
            local_rank_id=rank,
            world_size=world_size,
            balance_mode=balance_mode)

        expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=npu_balanced_expert_ids,
            x_active_mask=npu_balanced_active_mask,
            expert_scales=expert_scales,
            group_ep=ep_hcomm_info,
            group_tp=tp_hcomm_info,
            ep_world_size=ep_world_size,
            tp_world_size=tp_world_size,
            ep_rank_id=rank // tp_world_size,
            tp_rank_id=rank % tp_world_size,
            expert_shard_type=0,
            shared_expert_rank_num=shared_expert_rank_num,
            moe_expert_num=phy_ep_size,
            scales=scales,
            quant_mode=quant_mode,
            global_bs=globalBS)
        if is_quant:
            expand_x = expand_x.to(input_dtype)
        x = torch_npu.npu_moe_distribute_combine_v2(expand_x=expand_x,
                                                 expert_ids=npu_balanced_expert_ids,
                                                 x_active_mask=npu_balanced_active_mask,
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
                                                 global_bs=globalBS)
        print(f'rank {rank} epid {rank // tp_world_size} tpid {rank % tp_world_size} npu finished! \n')
    
    if __name__ == "__main__":
        print(f"BS={BS}")
        print(f"global_bs={globalBS}")
        print(f"shared_expert_rank_num={shared_expert_rank_num}")
        print(f"moe_expert_num={moe_expert_num}")
        print(f"K={K}")
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
    # Modify graph_type to support static and dynamic graphs
    import os
    import torch
    import random
    import torch_npu
    import torchair
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    # Control mode
    quant_mode = 2
    is_dispatch_scales = True
    input_dtype = torch.bfloat16
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)
    shared_expert_rank_num = 2
    moe_expert_num = 14
    bs = 8
    h = 7168
    k = 8
    random_seed = 0
    tp_world_size = 1
    ep_world_size = int(world_size / tp_world_size)
    moe_rank_num = ep_world_size - shared_expert_rank_num
    local_moe_expert_num = moe_expert_num // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)
    is_pruning = 1
    balance_mode = 1

    class MOE_DISTRIBUTE_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x, expert_ids, group_ep, group_tp, ep_world_size, tp_world_size, ep_rank_id, tp_rank_id,
                    expert_shard_type, shared_expert_rank_num, moe_expert_num, scales, quant_mode, global_bs, expert_scales, eplb_table, pruning_threshold, active_mask, balance_mode):
            balanced_expert_ids, balanced_active_mask = torch_npu.npu_moe_update_expert(expert_ids=expert_ids,
                                                                                        eplb_table=eplb_table,
                                                                                        expert_scales=expert_scales if is_pruning else None,
                                                                                        pruning_threshold=pruning_threshold,
                                                                                        active_mask=active_mask,
                                                                                        local_rank_id=ep_rank_id,
                                                                                        world_size=ep_world_size,
                                                                                        balance_mode=balance_mode)
            output_dispatch_npu = torch_npu.npu_moe_distribute_dispatch_v2(x=x,
                                                    expert_ids=balanced_expert_ids,
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
            expand_x_npu, _, assist_info_for_combine_npu, _, ep_recv_counts_npu, tp_recv_counts_npu, expand_scales = output_dispatch_npu
            if expand_x_npu.dtype == torch.int8:
                expand_x_npu = expand_x_npu.to(input_dtype)
            output_combine_npu = torch_npu.npu_moe_distribute_combine_v2(expand_x=expand_x_npu,
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
        expert_ids = gen_unique_topk_array(0, moe_expert_num, bs, k).astype(np.int32)
        expert_ids = torch.from_numpy(expert_ids).npu()
        eplb_table = np.array([[1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13]]).astype(np.int32)
        eplb_table = torch.from_numpy(eplb_table).npu()
        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()

        pruning_threshold = None
        active_mask = None
        if is_pruning:
            expert_scales_temp = -np.sort(-np.random.uniform(low=0, high=0.25, size=(bs, k)), axis=1)
            pruning_threshold_temp = np.random.uniform(low=0, high=0.15, size=(1, k))
            num_true = np.random.randint(0, bs + 1)
            active_mask_temp = np.concatenate([np.ones(num_true, dtype=bool), np.zeros(bs - num_true, dtype=bool)])
            expert_scales_tensor = torch.from_numpy(expert_scales_temp).to(torch.float32)
            pruning_threshold_tensor = torch.from_numpy(pruning_threshold_temp).to(torch.float32)
            active_mask_tensor = torch.from_numpy(active_mask_temp).to(torch.bool)
            expert_scales = expert_scales_tensor.npu()
            pruning_threshold = pruning_threshold_tensor.npu()
            active_mask = active_mask_tensor.npu()
        
        scales_shape = (1 + moe_expert_num, h) if shared_expert_rank_num else (moe_expert_num, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None
    
        model = MOE_DISTRIBUTE_GRAPH_Model()
        model = model.npu()
        npu_backend = torchair.get_npu_backend()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        output = model.forward(x, expert_ids, ep_hcomm_info, tp_hcomm_info, ep_world_size, tp_world_size,
                            rank // tp_world_size, rank % tp_world_size, 0, shared_expert_rank_num, moe_expert_num, scales,
                            quant_mode, globalBS, expert_scales, eplb_table, pruning_threshold, active_mask, balance_mode)
        torch.npu.synchronize()
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
        print("run npu success")
    ```
