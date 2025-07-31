# torch_npu.npu_moe_distribute_dispatch_teardown

## 支持的产品型号

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

## 功能说明

算子功能：
接收MOE层EP（Expert Parallelism）域的AllToAllV通信发过来的数据，数据发送端由`torch_npu.npu_moe_distribute_dispatch_setup`完成，本接口内完成通信状态确认和数据整理。

- 注意该接口必须与 `torch_npu.npu_moe_distribute_dispatch_setup`、`torch_npu.npu_moe_distribute_combine_setup`、`torch_npu.npu_moe_distribute_combine_teardown` 配套使用

详细说明请参考以下参数说明。

## 函数原型

* `torch_npu.npu_moe_distribute_dispatch_teardown(Tensor x, Tensor y, Tensor expert_ids, Tensor comm_cmd_info, str group_ep, int ep_world_size, int ep_rank_id, int moe_expert_num, int expert_shard_type=0, int shared_expert_num=1, int shared_expert_rank_num=0, int quant_mode=0, int global_bs=0, int expert_token_nums_type=1, int comm_type=0, str comm_alg="") -> (Tensor, Tensor, Tensor, Tensor)`

## torch\_npu.npu\_moe\_distribute\_dispatch\_teardown
## 参数说明

-   x：Tensor类型，表示本卡发送的token数据，Device侧的Tensor。要求为2D的Tensor，shape为\(BS, H\)，其中BS为batch size，H为hidden size（隐藏层大小）。数据类型支持bfloat16、float16，数据格式为ND，支持非连续的Tensor。
-   y：Tensor类型，表示本卡发送的token数据，Device侧的aclTensor。要求为一个2D的Tensor，shape为 \(BS \* \(K \+ sharedExpertNum\), token\_msg\_size\)，其中BS为batch size，H为hidden size，即隐藏层大小，token\_msg\_size表示每个token的在数据通信时的维度信息，非量化场景下token\_msg\_size= Align256\(H\)；量化场景下token\_msg\_size = Align512\(Align32\(H\) + 4\)，其中AlignN\(x\) = \(\(x \+ N \- 1\) / N\) \* N。每个数据类型支持float16、bfloat16、int8，数据格式为ND，支持非连续的Tensor。
-   expert\_ids：Tensor类型，表示每个token的topK个专家索引，Device侧的Tensor。要求为2D的Tensor，shape为\(BS, K\)，数据类型支持int32，数据格式为ND，支持非连续的Tensor。
-   comm\_cmd\_info：Tensor类型，通信的cmd信息，要求为1D的Tensor。数据类型int32，数据格式为ND，支持非连续的Tensor。
-   group\_ep：string类型，EP通信域名称，专家并行的通信域。字符串长度范围为\[1, 128\)。
-   ep\_world\_size：int类型，EP通信域size。l取值范围\[2, 384\]。
-   ep\_rank\_id：int类型，EP域本卡ID，取值范围\[0, ep\_world\_size\)。同一EP通信域内各卡的ep\_rank\_id不可重复。
-   moe\_expert\_num：int类型，MoE专家数量，取值范围\(0, 512\]，且需满足moe\_expert\_num % \(ep\_world\_size - shared\_expert\_rank\_num\) == 0。
-   expert\_shard\_type：int类型，表示共享专家卡分布类型。当前仅支持传0，表示共享专家卡位于MoE专家卡之前。
-   shared\_expert\_num：int类型，表示共享专家数量。取值范围[0, 4]，0表示无共享专家。单个共享专家可部署至多张卡。
-   shared\_expert\_rank\_num：int类型，表示共享专家卡数量。取值范围\[0, ep\_world\_size / 2\]，shared\_expert\_rank\_num % shared\_expert\_num = 0。
-   quant\_mode：int类型，量化模式。支持0（非量化）、2（动态量化）。
-   global\_bs：int类型，EP域全局batch size。当各rank的BS一致时，global\_bs = BS \* ep\_world\_size或global\_bs = 0；当BS不一致时，global\_bs = max\_bs \* ep\_world\_size，其中max\_bs为单卡BS最大值。
-   expert\_token\_nums\_type：int类型，输出expert\_token\_nums中值的语义类型。支持0：expert\_token\_nums中的输出为每个专家处理的token数的前缀和，1：expert\_token\_nums中的输出为每个专家处理的token数量。
-   comm\_type：int类型，通信方案选择。仅支持取值为0表示AICPU-SDMA方案。
-   comm\_alg：string类型，通信亲和内存布局算法。预留字段，当前版本不支持，传空串即可。

## 输出说明

-   expand\_x：Tensor类型，根据expertIds进行扩展过的token特征，要求为一个2D的Tensor，shape为 \(A, H\)，数据类型支持float16、bfloat16、int8，数据格式为ND，支持非连续的Tensor。
-   dynamic\_scales：Tensor类型，数据类型FLOAT32，要求为一个1D的Tensor，shape为 \(A, \)，数据格式为ND，支持非连续的Tensor。当quantMode为2时，才有该输出。
-   assist\_info\_for\_combine：Tensor类型，表示给同一专家发送的token个数，对应Combine系列算子中的assist\_info\_for\_combine，要求是一个1D的Tensor。数据类型支持INT32，数据格式为ND，支持非连续的Tensor。
-   expert\_token\_nums：Tensor类型，表示每个专家收到的token个数，数据类型INT64，要求为一个1D的Tensor，shape为 \(local_expert_num, \)，数据格式为ND，支持非连续的Tensor。

## 约束说明

-   `torch_npu.npu_moe_distribute_dispatch_teardown` 接口与 `torch_npu.npu_moe_distribute_dispatch_setup`、`torch_npu.npu_moe_distribute_combine_setup`、`torch_npu.npu_moe_distribute_combine_teardown`系列算子接口必须配套使用。

-   调用接口过程中使用的 `group_ep`、`ep_world_size`、`expert_shard_type`、`shared_expert_num`、`shared_expert_rank_num`、`global_bs` 参数取值需所有卡保持一致，网络中不同层中也需保持一致，且与 `torch_npu.npu_moe_distribute_dispatch_setup`、`torch_npu.npu_moe_distribute_combine_setup`、`torch_npu.npu_moe_distribute_combine_teardown` 算子对应参数保持一致。

-   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。

-   参数说明里shape格式说明：
    - A：表示本卡可能接收的最大token数量，取值范围如下：
        - 对于共享专家，当 `global_bs` 为0时，需满足 `A = BS * ep_world_size * shared_expert_num / shared_expert_rank_num`；当 `global_bs` 非0时，需满足 `A = global_bs * shared_expert_num / shared_expert_rank_num`。
        - 对于MoE专家，当 `global_bs` 为0时，需满足 `A >= BS * ep_world_size * min(local_expert_num, K)`；当 `global_bs` 非0时，需满足 `A >= global_bs * min(local_expert_num, K)`。
    - H：表示hidden size（隐藏层大小），取值范围 \[1024, 8192\]。
    - BS：表示batch sequence size（本卡最终输出的token数量），取值范围 0 < BS ≤ 512。
    - K：表示选取topK个专家，取值范围 0 < K ≤ 16 且满足 0 < K ≤ `moe_expert_num`。
    - local_expert_num：表示本卡专家数量。
        - 对于共享专家卡，`local_expert_num = 1`。
        - 对于MoE专家卡，`local_expert_num = moe_expert_num / (ep_world_size - shared_expert_rank_num)`。
  
-   HCCL_BUFFSIZE：
    调用本接口前需检查HCCL_BUFFSIZE环境变量取值是否合理。该环境变量表示单个通信域占用内存大小（单位MB），默认为200MB（未配置时）。要求满足：
    ```
    ≥ 2 * (local_expert_num * max_bs * ep_world_size * Align512(Align32(2 * H) + 44) + (K + shared_expert_num) * max_bs * Align512(2 * H))
    ```
    其中 `AlignN(x) = ((x + N - 1) / N) * N`，local_expert_num需使用MoE专家卡的本卡专家数。

-   通信域使用约束：
    - 一个模型中的`torch_npu.npu_moe_distribute_dispatch_setup`、`torch_npu.npu_moe_distribute_dispatch_teardown`、`torch_npu.npu_moe_distribute_combine_setup`、`torch_npu.npu_moe_distribute_combine_teardown`算子仅支持相同EP通信域，且该通信域中不允许有其他算子。


## 调用示例

-   单算子模式调用

    ```python
    import os
    import unittest
    import numpy as np
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp

    # 控制模式
    quant_mode = 2               # 0为非量化，2为动态量化
    comm_quant_mode = 0          # combine的通信量化参数
    is_dispatch_scales = True    # 动态量化可选择是否传scales
    input_dtype = torch.bfloat16 # 输出dtype
    sharedExpertRankNum = 0      # 共享专家数
    moeExpertNum = 16            # moe专家数
    bs = 64                      # token数量
    h = 1024                     # 每个token的长度
    k = 8
    random_seed = 0
    ep_world_size = 16
    globalBS = bs * ep_world_size
    is_quant = (quant_mode > 0)


    def gen_unique_topk_array(low, high):
        array = []
        for _ in range(bs):
            top_idx = list(np.arange(low, high, dtype=np.int32))
            np.random.shuffle(top_idx)
            array.append(top_idx[0:k])
        return np.array(array)


    def moe_distribute_dispatch_combine_sdma_(rank, c2p, p2c):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + "127.0.0.1" + ':' + '50001'
        dist.init_process_group(backend='hccl', world_size=ep_world_size, rank=rank, init_method=init_method)
        ep_group = dist.new_group(backend="hccl", ranks=range(ep_world_size))
        ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

        # 创建输入tensor
        np.random.seed(random_seed)
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = gen_unique_topk_array(0, moeExpertNum).astype(np.int32)
        expert_ids = torch.from_numpy(expert_ids).npu()

        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + moeExpertNum, h) if sharedExpertRankNum == 0 else (moeExpertNum, h)
        if is_quant and is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None

        torch.npu.synchronize()
        y, expand_idx, comm_cmd_info = torch_npu.npu_moe_distribute_dispatch_setup(
            x=x,
            expert_ids=expert_ids,
            group_ep=ep_hcomm_info,
            ep_world_size=ep_world_size,
            ep_rank_id=rank,
            moe_expert_num=moeExpertNum,
            expert_shard_type=0,
            shared_expert_rank_num=sharedExpertRankNum,
            scales=scales,
            quant_mode=quant_mode,
            global_bs=globalBS)
        torch.npu.synchronize()
        expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums = torch_npu.npu_moe_distribute_dispatch_teardown(
            x=x, 
            y=y, 
            expert_ids=expert_ids,
            comm_cmd_info=comm_cmd_info,
            group_ep=ep_hcomm_info,
            ep_world_size=ep_world_size,
            ep_rank_id=rank,
            moe_expert_num=moeExpertNum,
            expert_shard_type=0,
            shared_expert_rank_num=sharedExpertRankNum,
            quant_mode=quant_mode,
            global_bs=globalBS)
        if is_quant:
            expand_x = expand_x.to(input_dtype)
        torch.npu.synchronize()
        quant_expand_x, comm_cmd_info = torch_npu.npu_moe_distribute_combine_setup(
            expand_x=expand_x,  
            expert_ids=expert_ids,
            assist_info_for_combine=assist_info_for_combine,
            group_ep=ep_hcomm_info,
            ep_world_size=ep_world_size,
            ep_rank_id=rank,
            moe_expert_num=moeExpertNum,
            expert_shard_type=0,
            shared_expert_rank_num=sharedExpertRankNum,
            comm_quant_mode=comm_quant_mode,
            global_bs=globalBS)
        torch.npu.synchronize()
        out = torch_npu.npu_moe_distribute_combine_teardown(
            expand_x=expand_x, 
            quant_expand_x=quant_expand_x,
            expert_ids=expert_ids,
            expand_idx=expand_idx, 
            expert_scales=expert_scales,
            comm_cmd_info=comm_cmd_info,
            group_ep=ep_hcomm_info,
            ep_world_size=ep_world_size,
            ep_rank_id=rank,
            moe_expert_num=moeExpertNum,
            expert_shard_type=0,
            shared_expert_rank_num=sharedExpertRankNum,
            comm_quant_mode=comm_quant_mode,
            global_bs=globalBS)
        c2p.put((rank, out.cpu()))
        p2c.get()

    if __name__ == '__main__':
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(ep_world_size)
        p2c = ctx.Queue(ep_world_size)
        p_list = []
        for rank in range(ep_world_size):
            p = ctx.Process(target=moe_distribute_dispatch_combine_sdma_, args=(rank, c2p, p2c))
            p.start()
            p_list.append(p)
        for _ in range(ep_world_size):
            rank_id, out = c2p.get()
            print("recv rank", rank_id, "data success")
        for _ in range(ep_world_size):
            p2c.put(0)
        for p in p_list:
            p.join()
    ```


