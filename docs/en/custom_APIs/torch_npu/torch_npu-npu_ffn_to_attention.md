# torch\_npu.npu\_ffn\_to\_attention<a name="en-us_topic_0000002343094193"></a>

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |

## Function<a name="en-us_topic_0000002203575833_section14441124184110"></a>

  Sends token data from the FFN node to the Attention node.

## Prototype<a name="en-us_topic_0000002203575833_section45077510411"></a>

```python
torch_npu.npu_ffn_to_attention(x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num, group, world_size,token_info_table_shape, token_data_shape, *, attn_rank_table=None) -> ()
```

## Parameters<a name="en-us_topic_0000002203575833_section112637109429"></a>

- **`x`** (`Tensor`): Required. Token data used for computation and sent to other ranks based on `session_ids`. This parameter must be 2D with shape `(Y, H)`, indicating that there are `Y` tokens. The data type can be `bfloat16` or `float16`. The data layout can be ND. Non-contiguous tensors are supported.
- **`session_ids`** (`Tensor`): Required. Index of the Attention Worker node for each token, which determines the Attention Worker nodes to which each token is sent. This parameter must be 1D with shape `(Y,)`. The data type can be `int32`. The data layout can be ND. Non-contiguous tensors are supported. The value range of elements in the tensor is [0, attnRankNum - 1].
- **`micro_batch_ids`** (`Tensor`): Required. microBatch index for each token. This parameter must be 1D with shape `(Y,)`. The data type can be `int32`. The data layout can be ND. Non-contiguous tensors are supported. The value range of elements in the tensor is [0, `MicroBatchNum` - 1].
- **`token_ids`** (`Tensor`): Required. microBatch index for each token. This parameter must be 1D with shape `(Y,)`. The data type can be `int32`. The data layout can be ND. Non-contiguous tensors are supported. The value range of elements in the tensor is [0, `BS` - 1].
- **`expert_offsets`** (`Tensor`): Required. Index of `PerTokenExpertNum` in `token_info_table_shape` for each token. This parameter must be 1D with shape `(Y,)`. The data type can be `int32`. The data layout can be ND. Non-contiguous tensors are supported. The value range of elements in the tensor is [0, `ExpertNumPerToken` - 1].
- **`actual_token_num`** (`Tensor`): Required. Total number of tokens sent by the current rank. This parameter must be 1D with shape `(1,)`. The data type can be `int64`. The data layout can be ND. Non-contiguous tensors are supported. The value range of elements in the tensor is [0, `Y`].
- **`group`** (`str`): Required. Communication domain name used for expert parallelism. The string length range is [1, 128).
- **`world_size`** (`int64`): Required. Size of the communication domain. The value range is [2, 768].
- **`token_info_table_shape`** (`List(int)`): Required. Size of the token information list. It includes the microBatch size (`MicroBatchNum`), batch size (`Bs`), and the number of experts corresponding to each token (`ExpertNumPerToken`).
- **`token_data_shape`** (`List(int)`): Required. Size of the token information list. It includes the microBatch size (`MicroBatchNum`), batch size (`Bs`), the number of experts corresponding to each token (`ExpertNumPerToken`), and the length of tokens and scales (`HS`).
- **`attn_rank_table`** (`Tensor`): Optional. Maps the rank ID corresponding to each Attention Worker. This parameter must be 1D with shape `(Y,)`. The data type can be `int32`. The data layout can be ND. Non-contiguous tensors are supported. The value range of elements in the tensor is [0, attnRankNum - 1].

## Constraints<a name="en-us_topic_0000002203575833_section12345537164214"></a>

- The values of the `group`, `world_size`, `token_info_table_shape`, `token_data_shape`, and `HCCL_BUFFSIZE` parameters used during API execution must be identical across all ranks and across different layers in the network.
- Atlas A3 training products/Atlas A3 inference products: In this scenario, a single rank contains dual dies. Therefore, "this rank" in the parameter description indicates a single die.

- Variables used in input parameter tensor shapes:
    - `Y`: Maximum number of tokens to be distributed by the current rank.

    - `BS`: Number of tokens sent on each Attention node. The value range is 0 < `BS` ≤ 512.

    - `H`: Size of the hidden layer. The value range is 1152 ≤ `H` ≤ 8320.

    - `HS`: Size of the hidden and scale layers. The value range is 1024 ≤ `HS` ≤ 8192.

    - `MicroBatchNum`: Size of the microBatch. Currently, only `MicroBatchNum = 1` is supported.

    - `ExpertNumPerToken`: Number of experts sent corresponding to each token. The calculation formula is `ExpertNumPerToken` = `K` + `sharedExpertNum`.

    - `K`: Number of top-K experts selected. The value range is 0 < `K` ≤ 16.

    - `ffnRankNum` (`int`): Optional. Number of ranks selected as FFN workers. The value range is `0 < ffnRankNum < world_size`.

    - `attnRankNum`: Number of ranks selected as AttnWorkers. The value range is 0 < `attnRankNum` < `world_size`.

    - `sharedExpertNum`: Number of shared experts (a shared expert can be replicated and deployed across multiple FFN ranks). The value range is `0 ≤ sharedExpertNum ≤ 4`.

- Communication domain usage constraints:
    - No other operators are allowed in the communication domain of the `FFNToAttention` operator.

## Example<a name="en-us_topic_0000002203575833_section14459801435"></a>

- Single-operator call

    ```python
    import os
    import math
    import torch
    import random
    import torch_npu
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import time

    # Control mode
    input_dtype = torch.bfloat16  # Output dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # Number of dies per host
    micro_batch_num = 1
    bs = 8  # Number of tokens
    h = 7168  # Length of each token
    scale = 128
    hs = h + scale 
    k = 4
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
    token_info_table_shape = [micro_batch_num, bs, expert_num_per_token]
    token_data_table_shape = [micro_batch_num, bs, expert_num_per_token, hs]
    Y = int(math.ceil(micro_batch_num * bs * attention_worker_num * expert_num_per_token / ffn_worker_num))


    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info

    def ffn2attn_get_kwargs(
        x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num,
        attn_rank_table, group, world_size, token_info_table_shape, token_data_shape
    ):
        x = x.to(input_dtype).npu()
        session_ids = session_ids.to(torch.int32).npu()
        micro_batch_ids = micro_batch_ids.to(torch.int32).npu()
        token_ids = token_ids.to(torch.int32).npu()
        expert_offsets = expert_offsets.to(torch.int32).npu()
        actual_token_num = actual_token_num.to(torch.int64).npu()
        attn_rank_table = attn_rank_table.to(torch.int32).npu()
        
        return {
            'x':x,
            'session_ids':session_ids,
            'micro_batch_ids':micro_batch_ids,
            'token_ids':token_ids,
            'expert_offsets':expert_offsets,
            'actual_token_num':actual_token_num,
            'attn_rank_table':attn_rank_table,
            'group':group,
            'world_size':world_size,
            'token_info_table_shape':token_info_table_shape,
            'token_data_shape':token_data_shape
        }

    def set_windows(rank, comm_group, hcomm_info):
        if attention_worker_num <= rank < world_size:
            target_ranks = list(range(attention_worker_num))
        else:
            target_ranks = list(range(attention_worker_num, world_size))
        window_size = 1024 * 1024 * 200
        comm_group._get_backend(torch.device('npu'))._window_register_and_exchange(window_size,target_ranks)

    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        rank_list = list(range(world_size))
        comm_group = dist.new_group(backend="hccl", ranks=rank_list)
        hcomm_info = get_hcomm_info(rank, comm_group)
        set_windows(rank, comm_group, hcomm_info)

        # Create input tensors
        x = torch.randn((Y, h), dtype=input_dtype, device='npu')
        base_ids = torch.arange(Y, dtype=torch.int32, device='npu')
        session_ids = torch.fmod(base_ids, attention_worker_num)
        micro_batch_ids = torch.fmod(base_ids, micro_batch_num)
        token_ids = torch.fmod(base_ids, bs)
        expert_offsets = torch.fmod(base_ids, expert_num_per_token)
        token_num = torch.tensor([Y], dtype=torch.int64, device='npu')
        attn_rank_table = torch.arange(attention_worker_num,dtype = torch.int32)

        ffn2attn_kwargs = ffn2attn_get_kwargs(
            x=x,
            session_ids = session_ids,
            micro_batch_ids = micro_batch_ids,
            token_ids = token_ids,
            expert_offsets = expert_offsets,
            actual_token_num = token_num,
            attn_rank_table = attn_rank_table,
            group=hcomm_info,
            world_size=world_size,
            token_info_table_shape = token_info_table_shape,
            token_data_shape = token_data_table_shape
        )
        if rank >= attention_worker_num:
            torch_npu.npu_ffn_to_attention(**ffn2attn_kwargs)

        else:
            time.sleep(10)
        print(f'rank {rank} npu finished! \n')


    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"Y={Y}")
        print(f"k={k}")
        print(f"ffn_worker_num={ffn_worker_num}")
        print(f"attention_worker_num={attention_worker_num}")

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
    # Only static graphs are supported.
    import os
    import math
    import torch
    import random
    import torch_npu
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import torchair
    import time

    # Control mode
    input_dtype = torch.bfloat16  # Output dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # Number of dies per host
    micro_batch_num = 1
    bs = 8  # Number of tokens
    h = 7168  # Length of each token
    scale = 128
    hs = h + scale 
    k = 4
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
    token_info_table_shape = [micro_batch_num, bs, expert_num_per_token]
    token_data_table_shape = [micro_batch_num, bs, expert_num_per_token, hs]
    Y = int(math.ceil(micro_batch_num * bs * attention_worker_num * expert_num_per_token / ffn_worker_num))


    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info

    class FFN_TO_ATTENTION_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(
                self, x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num, attn_rank_table, group, world_size, token_info_table_shape, 
                token_data_shape
            ):
            output = torch_npu.npu_ffn_to_attention(
                x=x,
                session_ids=session_ids,
                micro_batch_ids=micro_batch_ids,
                token_ids=token_ids,
                expert_offsets=expert_offsets,
                actual_token_num=actual_token_num,
                attn_rank_table=attn_rank_table,
                group=group,
                world_size=world_size,
                token_info_table_shape=token_info_table_shape,
                token_data_shape=token_data_shape
            )
            return output


    def set_windows(rank, comm_group, hcomm_info):
        if attention_worker_num <= rank < world_size:
            target_ranks = list(range(attention_worker_num))
        else:
            target_ranks = list(range(attention_worker_num, world_size))
        window_size = 1024 * 1024 * 200
        comm_group._get_backend(torch.device('npu'))._window_register_and_exchange(window_size,target_ranks)

    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        rank_list = list(range(world_size))
        comm_group = dist.new_group(backend="hccl", ranks=rank_list)
        hcomm_info = get_hcomm_info(rank, comm_group)
        set_windows(rank, comm_group, hcomm_info)

        # Create input tensors
        x = torch.randn((Y, h), dtype=input_dtype, device='npu')
        base_ids = torch.arange(Y, dtype=torch.int32, device='npu')
        session_ids = torch.fmod(base_ids, attention_worker_num)
        micro_batch_ids = torch.fmod(base_ids, micro_batch_num)
        token_ids = torch.fmod(base_ids, bs)
        expert_offsets = torch.fmod(base_ids, expert_num_per_token)
        token_num = torch.tensor([Y], dtype=torch.int64, device='npu')
        attn_rank_table = torch.arange(attention_worker_num,dtype = torch.int32)


        if rank >= attention_worker_num:
            model = FFN_TO_ATTENTION_GRAPH_Model()
            model = model.npu()
            npu_backend = torchair.get_npu_backend()
            print ('Traditional graph input mode, static shape + online compilation scenario')
            model = torch.compile(model, backend = npu_backend, dynamic = False)
            _ = model.forward(
                x, session_ids, micro_batch_ids, token_ids, expert_offsets, token_num,
                attn_rank_table, hcomm_info, world_size, token_info_table_shape, 
                token_data_table_shape)

        else:
            time.sleep(10)
        print(f'rank {rank} npu finished! \n')


    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"Y={Y}")
        print(f"k={k}")
        print(f"ffn_worker_num={ffn_worker_num}")
        print(f"attention_worker_num={attention_worker_num}")

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
