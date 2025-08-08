# torch_npu.npu_moe_eplb_update_expert<a name="ZH-CN_TOPIC_0000002350725100"></a>

## 产品支持情况<a name="zh-cn_topic_0000002366611733_section8593133131718"></a>

<a name="zh-cn_topic_0000002366611733_table1659316316174"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002366611733_row2059343171716"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002366611733_p125930301711"><a name="zh-cn_topic_0000002366611733_p125930301711"></a><a name="zh-cn_topic_0000002366611733_p125930301711"></a><span id="zh-cn_topic_0000002366611733_ph12593183191719"><a name="zh-cn_topic_0000002366611733_ph12593183191719"></a><a name="zh-cn_topic_0000002366611733_ph12593183191719"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002366611733_p18593639173"><a name="zh-cn_topic_0000002366611733_p18593639173"></a><a name="zh-cn_topic_0000002366611733_p18593639173"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002366611733_row294304412306"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002366611733_p49437440302"><a name="zh-cn_topic_0000002366611733_p49437440302"></a><a name="zh-cn_topic_0000002366611733_p49437440302"></a><span id="zh-cn_topic_0000002366611733_ph19280164145411"><a name="zh-cn_topic_0000002366611733_ph19280164145411"></a><a name="zh-cn_topic_0000002366611733_ph19280164145411"></a><term id="zh-cn_topic_0000002366611733_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002366611733_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002366611733_zh-cn_topic_0000001312391781_term1253731311225"></a><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002366611733_p8877121915317"><a name="zh-cn_topic_0000002366611733_p8877121915317"></a><a name="zh-cn_topic_0000002366611733_p8877121915317"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0000002366611733_section14441124184110"></a>

-   算子功能：为了解决负载不均衡的场景，MoE网络中常用EPLB（Expert Parallelism Load Balancer）算法进行冗余专家部署，一个逻辑专家在多个卡上都有实例部署（即有多个物理专家），在这种场景下，MoeEPLBUpdateExpert算子可以完成每个token的topK个专家逻辑专家号到物理专家实例号的映射。
-   计算公式：

    对于`expert_ids`中的第i个值，即第i个token：

    ![](figures/zh-cn_formulaimage_0000002350811180.png)

    -   当`eplb_table`\[tableOffset\]=1时

        ![](figures/zh-cn_formulaimage_0000002384541317.png)

    -   当`eplb_table`\[tableOffset\]\>1时

        ![](figures/zh-cn_formulaimage_0000002350982146.png)

## 函数原型<a name="zh-cn_topic_0000002366611733_section45077510411"></a>

```
torch_npu.npu_moe_eplb_update_expert(Tensor expert_ids, Tensor eplb_table, int local_rank_id, int world_size, *, int balance_mode=0) -> Tensor
```

## 参数说明<a name="zh-cn_topic_0000002366611733_section112637109429"></a>

-   **expert_ids**（`Tensor`）：必选参数，表示每个token的topK个专家索引，数据类型支持`int32`、`int64`。数据格式支持ND。维度支持2维，shape为$(BS, K)$，支持非连续的Tensor。
-   **eplb_table**（`Tensor`）：必选参数，表示逻辑专家到物理专家实例的映射表，请保证输入Tensor的值正确。`world_size`张卡，每张卡部署`place_per_rank`个路由专家实例，一共有`world_size` \* `place_per_rank`个实例。`eplb_table`的每行对应一个逻辑moe专家的部署策略，第一列为该逻辑专家部署的实例数count，值需大于等于1；每行第\[1, count\]列为对应的实例编号，取值范围为\[0, `world_size` * `place_per_rank`\)，有效的实例编号不可以重复。数据类型支持`int32`。数据格式支持ND。维度支持2维，shape为$(moe\_expert\_num, F)$，支持非连续的Tensor。
-   **local_rank_id**（`int`）：必选参数，表示本卡Id。数据类型支持int64。取值范围为\[0, `world_size`\)。同一个通信域中各卡的`local_rank_id`不重复。
-   **world_size**（`int`）：必选参数，表示通信域Size。取值范围为\[2, 384\]。
-   **balance_mode**（`int`）：**预留参数，暂未使用，使用默认值即可，默认值为0**。

## 返回值说明<a name="zh-cn_topic_0000002366611733_section22231435517"></a>

**balanced_expert_ids**（`Tensor`）：表示映射后每个token的topK个专家物理专家的实例编号，要求是一个2D的Tensor，shape为$(BS, K)$，数据类型、数据格式与`expert_ids`保持一致，不支持非连续的Tensor。

## 约束说明<a name="zh-cn_topic_0000002366611733_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。
-   该接口必须与`MoeDistributeDispatchV2`，`MoeDistributeCombineV2`或`MoeDistributeCombineAddRmsNorm`接口配合使用，调用顺序为`MoeEPLBUpdateExpertNum` -> `MoeDistributeDispatchV2` -> `MoeDistributeCombineV2`/`MoeDistributeCombineAddRmsNorm`。
-   参数里Shape使用的变量如下：
    -   `BS`：表示batch sequence size，即本卡最终输出的token数量，取值范围为0<BS≤512。
    -   `K`：表示选取topK个专家，取值范围为0< K ≤16同时满足0 < K ≤ `moe_expert_num`。
    -   `moe_expert_num`：表示MoE专家数，取值范围\(0, 512\]。
    -   `F`：表示输入映射表的列数，第一列为各行号对应MoE专家部署的实例个数（取值\> 0），后F-1列为该MoE专家部署的实例编号，取值范围\[0, `world_size` \* `moe_expert_per_rank`\)。
    -   所有卡部署的moe专家实例总数最多1024，即`place_per_rank` \* `world_size` ≤ 1024。
    -   每张卡部署的实例数需相同。

## 调用示例<a name="zh-cn_topic_0000002366611733_section14459801435"></a>

-   单算子模式调用

    ```python
    import os
    import torch
    import random
    import torch_npu
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    
    # 控制模式
    quant_mode = 2                       # 2为动态量化
    is_dispatch_scales = True            # 动态量化可选择是否传scales
    input_dtype = torch.bfloat16         # 输出dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # 每个host有几个die
    sharedExpertRankNum = 2                      # 共享专家数
    moeExpertNum = 14                            # moe专家数
    bs = 8                                       # token数量
    h = 7168                                     # 每个token的长度
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
            # 如果tp_world_size = 2，ep_world_size = 8，则为[[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
            ep_ranks = [x * tp_world_size + i for x in range(ep_world_size)]
            ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group_t = ep_group
                print(f"rank:{rank} ep_ranks:{ep_ranks}")
        for i in range(ep_world_size):
            # 如果tp_world_size = 2，ep_world_size = 8，则为[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
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
    
        # 创建输入tensor
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = gen_unique_topk_array(0, moeExpertNum, bs, k).astype(np.int32)
        expert_ids = torch.from_numpy(expert_ids).npu()
        eplb_table = np.array([[1, 0], [1, 1], [1, 2], [1, 3], [1, 4],  [1, 5],
        [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13]]).astype(np.int32)
        eplb_table = torch.from_numpy(eplb_table).npu()
    
        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + moeExpertNum, h) if sharedExpertRankNum else (moeExpertNum, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None
    
        balanced_expert_ids = torch_npu.npu_moe_eplb_update_expert(
            expert_ids=expert_ids,
            eplb_table=eplb_table,
            local_rank_id=rank // tp_world_size,
            world_size=ep_world_size,
            balance_mode=0)

        expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=balanced_expert_ids,
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
        x = torch_npu.npu_moe_distribute_combine_v2(expand_x=expand_x,
                                                 expert_ids=balanced_expert_ids,
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
            print("sharedExpertRankNum 不能大于 ep_world_size")
            exit(0)
    
        if sharedExpertRankNum > 0 and ep_world_size % sharedExpertRankNum != 0:
            print("ep_world_size 必须是 sharedExpertRankNum的整数倍")
            exit(0)
    
        if moeExpertNum % moe_rank_num != 0:
            print("moeExpertNum 必须是 moe_rank_num 的整数倍")
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

-   图模式调用

    ```python
    # 仅支持静态图
    import os
    import torch
    import random
    import torch_npu
    import torchair
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    
    # 控制模式
    quant_mode = 2                         # 2为动态量化
    is_dispatch_scales = True              # 动态量化可选择是否传scales
    input_dtype = torch.bfloat16           # 输出dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # 每个host有几个die
    sharedExpertRankNum = 2                      # 共享专家数
    moeExpertNum = 14                            # moe专家数
    bs = 8                                       # token数量
    h = 7168                                     # 每个token的长度
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
                    scales, quant_mode, global_bs, expert_scales, eplb_table, balance_mode):
            balanced_expert_ids = torch_npu.npu_moe_eplb_update_expert(expert_ids=expert_ids,
                                                        eplb_table=eplb_table,
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
    
        # 创建输入tensor
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = gen_unique_topk_array(0, moeExpertNum, bs, k).astype(np.int32)
        expert_ids = torch.from_numpy(expert_ids).npu()
        balance_mode = 0
        eplb_table = np.array([[1, 0], [1, 1], [1, 2], [1, 3], [1, 4],  [1, 5],
        [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13]]).astype(np.int32)
        eplb_table = torch.from_numpy(eplb_table).npu()

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
                               rank // tp_world_size,rank % tp_world_size, 0, sharedExpertRankNum, moeExpertNum, scales,
                               quant_mode, globalBS, expert_scales, eplb_table, balance_mode)
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
            print("sharedExpertRankNum 不能大于 ep_world_size")
            exit(0)
    
        if sharedExpertRankNum > 0 and ep_world_size % sharedExpertRankNum != 0:
            print("ep_world_size 必须是 sharedExpertRankNum的整数倍")
            exit(0)
    
        if moeExpertNum % moe_rank_num != 0:
            print("moeExpertNum 必须是 moe_rank_num 的整数倍")
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
