# torch\_npu.npu\_attention\_to\_ffn

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/release_notes/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E).

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |

## Function

Sends token data from the Attention node to the FFN node.

## Prototype

```python
torch_npu.npu_attention_to_ffn(x, session_id, micro_batch_id, layer_id, expert_ids, expert_rank_table, group, world_size, ffn_token_info_table_shape, ffn_token_data_shape, attn_token_info_table_shape, moe_expert_num, *, scales=None, active_mask=None, quant_mode=0, sync_flag=0, ffn_start_rank_id=0) -> ()
```

## Parameters

- **`x`** (`Tensor`): Required. Token data used for computation and sent to other ranks based on `expert_ids` and `expert_rank_table`. This parameter must be 3D with shape `(X, BS, H)`, representing $X$ micro-batches with $BS$ tokens in each micro-batch. The data type can be `bfloat16` or `float16`. The data layout can be ND. Non-contiguous tensors are supported.
- **`session_id`** (`Tensor`): Required. Current rank ID within the Attention domain. This parameter must be 1D with shape `(X,)`. The data type can be `int32`. The data layout is ND. Non-contiguous tensors are supported.
- **`micro_batch_id`** (`Tensor`): Required. ID of the current micro-batch group. This parameter must be 1D with shape `(X,)`. The data type can be `int32`. The data layout is ND. Non-contiguous tensors are supported.
- **`layer_id`** (`Tensor`): Required. Model layer ID. This parameter must be 1D with shape `(X,)`. The data type can be `int32`. The data layout is ND. Non-contiguous tensors are supported.
- **`expert_ids`** (`Tensor`): Required. Top-K expert indices for each token in each micro-batch group, which determines the destination experts for each token. This parameter must be 3D with shape `(X, BS, K)`. The data type can be `int32`. The data layout is ND. Non-contiguous tensors are supported. The value range of elements inside the tensor is [0, `moe_expert_num`), and the $K$ values within the identical row must be unique.
- **`expert_rank_table`** (`Tensor`): Required. Mapping table from expert IDs to the FFN rank expert deployment configurations for each micro-batch group. Ensure that the values are correct. This parameter must be 3D with shape `(L, shared_expert_num + moe_expert_num, M)`. The data type can be `int32`. The data layout can be ND. Non-contiguous tensors are supported.
- **`group`** (`str`): Required. Communication domain name used for expert parallelism. The string length range is [1, 128).
- **`world_size`** (`int`): Required. Size of the communication domain. The value range is [2, 768].
- **`ffn_token_info_table_shape`** (`List[int]`): Required. Shape of the token information table on the FFN rank. The list length must be 3, including the number of Attention nodes, the micro-batch size, and the shape size of the related communication transmission status information per token.
- **`ffn_token_data_shape`** (`List[int]`): Required. Shape of the token data table on the FFN rank. The list length must be 5, including the number of Attention nodes, the micro-batch size, the batch size, the number of experts each token needs to be sent to (including shared experts), and the length of a single token.
- **`attn_token_info_table_shape`** (`List[int]`): Required. Shape of the token information table on the Attention card. The list length must be 3, including the micro-batch size, the batch size, and the number of experts each token needs to be sent to (including shared experts).
- **`moe_expert_num`** (`int`): Required. Number of MoE experts. The value range is [1, 1024].
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`scales`** (`Tensor`): Optional. Weight of each expert. This parameter is currently not used in non-quantization scenarios. In dynamic quantization scenarios, it can be provided or omitted. If provided, this parameter must be 3D with shape `(L, shared_expert_num + moe_expert_num, H)`. The data type can be `float`. The data layout can be ND. Non-contiguous tensors are not supported. When `quant_mode` is `2`, `scales` can be provided. When `quant_mode` is `0`, `scales` must be `None`.
- **`active_mask`** (`Tensor`): Optional. Specifies whether a token participates in communication. This parameter must be 2D with shape `(X, BS)`. The data type can be `bool`. The data layout must be ND. Non-contiguous tensors are supported. `True` indicates that the corresponding token will participate in communication. All `True` values must precede `False`. For example, `{True, False, True}` is an invalid input. By default, all tokens participate in communication.
- **`quant_mode`** (`int`): Optional. Quantization mode. Valid values are `0` (non-quantization mode, default) or `2` (dynamic quantization mode).
- **`sync_flag`** (`int`): Optional. Enables synchronous or asynchronous execution. Valid values are `0` (default, synchronous) or `1` (asynchronous).
- **`ffn_start_rank_id`** (`int`): Optional. Starting rank ID of the FFN domain. The value range is [0, `world_size`). The default value is `0`.

## Return Values

- None.

## Constraints

- This API can be used in inference scenarios.
- This API supports static graph mode. Separated operators must be used together.
- The values of the `group`, `world_size`, and `moe_expert_num` parameters used during API execution must be identical across all ranks and across different layers in the network.
- Atlas A3 training products/Atlas A3 inference products: In this scenario, a single rank contains dual dies. Therefore, "this rank" in the parameter description indicates a single die.
- Variables used in parameter tensor shapes:
    - `X`: Micro-batch sequence size (number of token groups). In the current version, only `X = 1` is supported.

    - `H`: Hidden layer size. The value range is [1024, 8192].

    - `BS`: Batch sequence size. That is, the number of tokens output by the current rank. The value range is 0 < `BS` ≤ 512.

    - `K`: Number of top-K experts selected. The value range is `0 < K ≤ 16`, and the condition `0 < K ≤ moe_expert_num` must be satisfied.

    - `L`: Number of model layers. In the current version, only `L = 1` is supported.

    - `shared_expert_num`: Number of shared experts (a shared expert can be replicated and deployed across multiple FFN ranks). The value range is [0, 4].

- HCCL communication buffer size: Before calling this API, verify that the configured HCCL communication domain buffer size is reasonable. The unit is MB, and the default value is `200` MB if not configured.

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
    import time

    # Control mode
    quant_mode = 2  # 2 enables dynamic quantization
    sync_flag = 1   # 1 enables asynchronous execution
    is_mask = True  # Specifies whether to enable pruning
    is_attn2ffn_scales = True  # Specifies whether to pass scales for dynamic quantization
    input_dtype = torch.bfloat16  # Output data type
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # Number of dies per host
    micro_batch_num = 1
    X = 1
    L = 1
    bs = 8  # Number of tokens
    h = 7168  # Length of each token
    k = 4
    hs = h + 128
    random_seed = 0
    shared_expert_num = 1  # Number of shared experts
    rank_num_per_shared_expert = 1
    shared_ffn_rank_num = shared_expert_num * rank_num_per_shared_expert
    moe_expert_per_rank = 2  # Number of MOE experts per FFN rank
    moe_ffn_rank_num = 4    # Number of FFN ranks
    moe_expert_num = moe_ffn_rank_num * moe_expert_per_rank
    ffn_worker_num = moe_ffn_rank_num + shared_ffn_rank_num
    attention_worker_num = world_size - ffn_worker_num
    expert_num_per_token = k + shared_expert_num
    is_quant = (quant_mode > 0)
    ffn_token_info_table_shape = [attention_worker_num, micro_batch_num, 2 + bs * expert_num_per_token]
    ffn_token_data_shape = [attention_worker_num, micro_batch_num, bs, expert_num_per_token, hs if is_quant else h]
    attn_token_info_table_shape = [micro_batch_num, bs, expert_num_per_token]


    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info
    

    def set_windows(rank, comm_group, hcomm_info):
        if rank >= ffn_worker_num:
            # The current rank belongs to the second half and needs to communicate with the first moe_ffn_rank_num ranks
            target_ranks = list(range(ffn_worker_num))
        else:
            # The current rank belongs to the first half and needs to communicate with the subsequent ranks
            target_ranks = list(range(ffn_worker_num, world_size))
        window_size = 120 * 1024 * 200
        comm_group._get_backend(torch.device('npu')._window_register_and_exchange(window_size, target_ranks))


    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        rank_list = list(range(world_size))
        comm_group = dist.new_group(backend="hccl", ranks=rank_list)
        hcomm_info = get_hcomm_info(rank, comm_group)
        set_windows(rank, comm_group, hcomm_info)

        # Create input tensors
        x = torch.randn(X, bs, h, dtype=input_dtype).npu()
        session_id = torch.tensor([rank - ffn_worker_num], dtype=torch.int32).npu()
        micro_batch_id = torch.tensor([0], dtype=torch.int32).npu()
        layer_id = torch.tensor([0], dtype=torch.int32).npu()
        expert_ids = torch.tensor([[[5, 7, 1, 4],
                                [0, 2, 3, 5],
                                [6, 3, 1, 7],
                                [4, 0, 5, 7],
                                [0, 6, 1, 3],
                                [2, 5, 7, 6],
                                [1, 6, 2, 4],
                                [6, 4, 5, 0]]], dtype=torch.int32).npu()
        expert_rank_table = torch.tensor([[[4, 2, 4, 3, 7, 1, 3, 2, 5],
                                        [2, 2, 5, 1, 2, 0, 0, 0, 0],
                                        [3, 2, 5, 0, 0, 3, 7, 0, 0],
                                        [4, 1, 3, 0, 1, 2, 4, 3, 7],
                                        [4, 0, 0, 3, 6, 1, 3, 2, 5],
                                        [3, 3, 7, 2, 4, 1, 2, 0, 0],
                                        [2, 2, 5, 0, 0, 0, 0, 0, 0],
                                        [3, 3, 6, 2, 5, 3, 7, 0, 0],
                                        [1, 4, 8, 0, 0, 0, 0, 0, 0]]], dtype=torch.int32).npu()

        scales_shape = (L, shared_expert_num + moe_expert_num, h)
        if is_attn2ffn_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None
        
        if is_mask:
            active_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool).npu()
        else:
            active_mask = None

        if rank >= ffn_worker_num:
            torch_npu.npu_attention_to_ffn(
                x=x,
                session_id = session_id,
                micro_batch_id = micro_batch_id,
                layer_id = layer_id,
                expert_ids=expert_ids,
                expert_rank_table = expert_rank_table,
                group=hcomm_info,
                world_size=world_size,
                moe_expert_num=moe_expert_num,
                ffn_token_info_table_shape = ffn_token_info_table_shape,
                ffn_token_data_shape = ffn_token_data_shape,
                attn_token_info_table_shape = attn_token_info_table_shape,
                scales=scales,
                active_mask=active_mask,
                quant_mode=quant_mode,
                sync_flag=sync_flag)

            print(f'rank {rank} npu finished! \n')
        
        if rank < ffn_worker_num:
            time.sleep(10)


    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"shared_expert_num={shared_expert_num}", flush=True)
        print(f"moe_expert_num={moe_expert_num}")

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
    quant_mode = 2  # 2 enables dynamic quantization
    sync_flag = 1   # 1 enables asynchronous execution
    is_mask = True  # Specifies whether to enable pruning
    is_attn2ffn_scales = True  # Specifies whether to pass scales for dynamic quantization
    input_dtype = torch.bfloat16  # Output data type
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # Number of dies per host
    micro_batch_num = 1
    X = 1
    L = 1
    bs = 8  # Number of tokens
    h = 7168  # Length of each token
    k = 4
    hs = h +128
    random_seed = 0
    shared_expert_num = 1  # Number of shared experts
    rank_num_per_shared_expert = 1
    shared_ffn_rank_num = shared_expert_num * rank_num_per_shared_expert
    moe_expert_per_rank = 2  # Number of MOE experts per FFN rank
    moe_ffn_rank_num = 4    # Number of FFN ranks
    moe_expert_num = moe_ffn_rank_num * moe_expert_per_rank
    ffn_worker_num = moe_ffn_rank_num + shared_ffn_rank_num
    attention_worker_num = world_size - ffn_worker_num
    expert_num_per_token = k + shared_expert_num
    is_quant = (quant_mode > 0)
    ffn_token_info_table_shape = [attention_worker_num, micro_batch_num, 2 + bs * expert_num_per_token]
    ffn_token_data_shape = [attention_worker_num, micro_batch_num, bs, expert_num_per_token, hs if is_quant else h]
    attn_token_info_table_shape = [micro_batch_num, bs, expert_num_per_token]


    class ATTENTION_TO_FFN_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, session_id, micro_batch_id, layer_id, expert_ids,
                expert_rank_table, hcomm_info, world_size, moe_expert_num,
                ffn_token_info_table_shape, ffn_token_data_shape,
                attn_token_info_table_shape, scales, active_mask, quant_mode,
                sync_flag):
            output = torch_npu.npu_attention_to_ffn(
                x=x,
                session_id = session_id,
                micro_batch_id = micro_batch_id,
                layer_id = layer_id,
                expert_ids=expert_ids,
                expert_rank_table = expert_rank_table,
                group=hcomm_info,
                world_size=world_size,
                moe_expert_num=moe_expert_num,
                ffn_token_info_table_shape = ffn_token_info_table_shape,
                ffn_token_data_shape = ffn_token_data_shape,
                attn_token_info_table_shape = attn_token_info_table_shape,
                scales=scales,
                active_mask=active_mask,
                quant_mode=quant_mode,
                sync_flag=sync_flag)
            
            return output


    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info
    

    def set_windows(rank, comm_group, hcomm_info):
        if rank >= ffn_worker_num:
            # The current rank belongs to the second half and needs to communicate with the first moe_ffn_rank_num ranks
            target_ranks = list(range(ffn_worker_num))
        else:
            # The current rank belongs to the first half and needs to communicate with the subsequent ranks
            target_ranks = list(range(ffn_worker_num, world_size))
        window_size = 120 * 1024 * 200
        comm_group._get_backend(torch.device('npu')._window_register_and_exchange(window_size, target_ranks))


    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(
            backend='hccl',
            rank=rank,
            world_size=world_size,
            init_method=f'tcp://{master_ip}:{port}'
        )
        rank_list = list(range(world_size))
        comm_group = dist.new_group(backend="hccl", ranks=rank_list)
        hcomm_info = get_hcomm_info(rank, comm_group)
        set_windows(rank, comm_group, hcomm_info)

        # Create input tensors
        x = torch.randn(X, bs, h, dtype=input_dtype).npu()
        session_id = torch.tensor([rank - ffn_worker_num], dtype=torch.int32).npu()
        micro_batch_id = torch.tensor([0], dtype=torch.int32).npu()
        layer_id = torch.tensor([0], dtype=torch.int32).npu()
        expert_ids = torch.tensor([[[5, 7, 1, 4],
                                [0, 2, 3, 5],
                                [6, 3, 1, 7],
                                [4, 0, 5, 7],
                                [0, 6, 1, 3],
                                [2, 5, 7, 6],
                                [1, 6, 2, 4],
                                [6, 4, 5, 0]]], dtype=torch.int32).npu()
        expert_rank_table = torch.tensor([[[4, 2, 4, 3, 7, 1, 3, 2, 5],
                                        [2, 2, 5, 1, 2, 0, 0, 0, 0],
                                        [3, 2, 5, 0, 0, 3, 7, 0, 0],
                                        [4, 1, 3, 0, 1, 2, 4, 3, 7],
                                        [4, 0, 0, 3, 6, 1, 3, 2, 5],
                                        [3, 3, 7, 2, 4, 1, 2, 0, 0],
                                        [2, 2, 5, 0, 0, 0, 0, 0, 0],
                                        [3, 3, 6, 2, 5, 3, 7, 0, 0],
                                        [1, 4, 8, 0, 0, 0, 0, 0, 0]]], dtype=torch.int32).npu()

        scales_shape = (L, shared_expert_num + moe_expert_num, h)
        if is_attn2ffn_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None
        
        if is_mask:
            active_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool).npu()
        else:
            active_mask = None

        if rank >= ffn_worker_num:
            model = ATTENTION_TO_FFN_GRAPH_Model()
            model = model.npu()
            npu_backend = torchair.get_npu_backend()
            model = torch.compile(model, backend=npu_backend, dynamic=False)
            _ = model.forward(
                x, session_id, micro_batch_id, layer_id, expert_ids, expert_rank_table,
                hcomm_info, world_size, moe_expert_num, ffn_token_info_table_shape, 
                ffn_token_data_shape, attn_token_info_table_shape, scales, active_mask, quant_mode, sync_flag
            )

            print(f'rank {rank} npu finished! \n')

        if rank < ffn_worker_num:
            time.sleep(10)


    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"shared_expert_num={shared_expert_num}", flush=True)
        print(f"moe_expert_num={moe_expert_num}")

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
