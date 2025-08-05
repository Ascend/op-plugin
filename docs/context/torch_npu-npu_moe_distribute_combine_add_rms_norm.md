# torch\_npu.npu\_moe\_distribute\_combine\_add\_rms\_norm<a name="ZH-CN_TOPIC_0000002384325441"></a>

## 产品支持情况<a name="zh-cn_topic_0000002322738573_section1369303644412"></a>

<a name="zh-cn_topic_0000002322738573_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002322738573_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002322738573_p1883113061818"><a name="zh-cn_topic_0000002322738573_p1883113061818"></a><a name="zh-cn_topic_0000002322738573_p1883113061818"></a><span id="zh-cn_topic_0000002322738573_ph24751558184613"><a name="zh-cn_topic_0000002322738573_ph24751558184613"></a><a name="zh-cn_topic_0000002322738573_ph24751558184613"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002322738573_p783113012187"><a name="zh-cn_topic_0000002322738573_p783113012187"></a><a name="zh-cn_topic_0000002322738573_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002322738573_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002322738573_p2098311377352"><a name="zh-cn_topic_0000002322738573_p2098311377352"></a><a name="zh-cn_topic_0000002322738573_p2098311377352"></a><span id="zh-cn_topic_0000002322738573_ph1719614396352"><a name="zh-cn_topic_0000002322738573_ph1719614396352"></a><a name="zh-cn_topic_0000002322738573_ph1719614396352"></a><term id="zh-cn_topic_0000002322738573_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002322738573_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002322738573_zh-cn_topic_0000001312391781_term1253731311225"></a><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002322738573_p7948163910184"><a name="zh-cn_topic_0000002322738573_p7948163910184"></a><a name="zh-cn_topic_0000002322738573_p7948163910184"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0000002322738573_section1470016430218"></a>

-   算子功能：完成moe\_distribute\_combine+add+rms\_norm融合。需与[torch_npu.npu_moe_distribute_dispatch](torch_npu-npu_moe_distribute_dispatch.md)配套使用，相当于按npu\_moe\_distribute\_dispatch算子收集数据的路径原路返回后对数据进行add\_rms\_norm操作。
-   计算公式：

    ![](figures/zh-cn_formulaimage_0000002340454445.png)

## 函数原型<a name="zh-cn_topic_0000002322738573_section470115437220"></a>

```
torch_npu.npu_moe_distribute_combine_add_rms_norm(Tensor expand_x, Tensor expert_ids, Tensor expand_idx, Tensor ep_send_counts, Tensor expert_scales, Tensor residual_x, Tensor gamma, str group_ep, int ep_world_size, int ep_rank_id, int moe_expert_num, *, Tensor? tp_send_counts=None, Tensor? x_active_mask=None, Tensor? activation_scale=None, Tensor? weight_scale=None, Tensor? group_list=None, Tensor? expand_scales=None, Tensor? shared_expert_x=None, str group_tp="", int tp_world_size=0, int tp_rank_id=0, int expert_shard_type=0, int shared_expert_num=1, int shared_expert_rank_num=0, int global_bs=0, int out_dtype=0, int comm_quant_mode=0, int group_list_type=0, float norm_eps=1e-06) -> (Tensor, Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002322738573_section187018431529"></a>

-   **expand\_x**（Tensor）：必选参数，根据expert\_ids进行扩展过的token特征，要求为2D的Tensor，shape为\(max\(tp\_world\_size, 1\) \*A, H\)，数据类型支持bfloat16，数据格式为ND，支持非连续的Tensor。
-   **expert\_ids**（Tensor）：必选参数，每个token的topK个专家索引，要求为2D的Tensor，shape为\(BS, K\)。数据类型支持int32，数据格式为ND，支持非连续的Tensor。对应torch\_npu.npu\_moe\_distribute\_dispatch的expert\_ids输入，张量里value取值范围为\[0, moe\_expert\_num\)，且同一行中的K个value不能重复。
-   **expand\_idx**（Tensor）：必选参数，表示给同一专家发送的token个数，要求是1D的Tensor，shape为\(A\*128, \)。数据类型支持int32，数据格式为ND，支持非连续的Tensor。对应torch\_npu.npu\_moe\_distribute\_dispatch的expand\_idx输出。
-   **ep\_send\_counts**（Tensor）：必选参数，表示本卡每个专家发给EP（Expert Parallelism）域每个卡的数据量，要求是1D的Tensor 。数据类型支持int32，数据格式为ND，支持非连续的Tensor。对应torch\_npu.npu\_moe\_distribute\_dispatch的ep\_recv\_counts输出。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求shape为\(ep\_world\_size\*max\(tp\_world\_size, 1\)\*local\_expert\_num, \)。

-   **expert\_scales**（Tensor）：必选参数，表示每个Token的topK个专家的权重，要求是2D的Tensor，shape为\(BS, K\)，其中共享专家不需要乘权重系数，直接相加即可。数据类型支持float，数据格式为ND，支持非连续的Tensor。
-   **residual\_x**（Tensor）：必选参数，表示处理后的token需要add的参数，要求是3D的Tensor，shape为\(BS, 1, H\)。数据类型支持bfloat16，数据格式为ND，支持非连续的Tensor。
-   **gamma**（Tensor）：必选参数，表示rms\_norm的权重，要求是1D的Tensor，shape为\(H, \)。数据类型支持bfloat16，数据格式为ND，支持非连续的Tensor。
-   **group\_ep**（str）：必选参数，EP通信域名称，专家并行的通信域。字符串长度范围为\[1, 128\)，不能和group\_tp相同。
-   **ep\_world\_size**（int）：必选参数，EP通信域size。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值支持\[2, 384\]。

-   **ep\_rank\_id**（int）：必选参数，EP通信域本卡ID，取值范围\[0, ep\_world\_size\)，同一个EP通信域中各卡的ep\_rank\_id不重复。
-   **moe\_expert\_num**（int）：必选参数，MoE专家数量，取值范围\[1, 512\]，并且满足moe\_expert\_num%\(ep\_world\_size-shared\_expert\_rank\_num\)=0。
-   **tp\_send\_counts**（Tensor）：可选参数，表示本卡每个专家发给TP（Tensor  Parallelism）通信域每个卡的数据量。对应torch\_npu.npu\_moe\_distribute\_dispatch的tp\_recv\_counts输出。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持TP通信域，要求是一个1D Tensor，shape为\(tp\_world\_size, \)，数据类型支持int32，数据格式要求为ND，支持非连续的Tensor。

-   **x\_active\_mask**（Tensor）：**预留参数，暂未使用，使用默认值即可。**
-   **activation\_scale**（Tensor）：**预留参数，暂未使用，使用默认值即可。**
-   **weight\_scale**（Tensor）：**预留参数，暂未使用，使用默认值即可。**
-   **group\_list**（Tensor）：**预留参数，暂未使用，使用默认值即可。**
-   **expand\_scales**（Tensor）：可选参数，对应torch\_npu.npu\_moe\_distribute\_dispatch的expand\_scales输出。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：暂不支持该参数，使用默认值即可。

-   **shared\_expert\_x**（Tensor）：可选参数，数据类型需与expand\_x保持一致。仅在共享专家卡数量shared\_expert\_rank\_num为0的场景下使用，表示共享专家token，在combine时需要加上。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型需与expand\_x保持一致，shape为\[BS, H\]。

-   **group\_tp**（str）：可选参数，TP通信域名称，数据并行的通信域。有TP域通信才需要传参。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当有TP域通信时，字符串长度范围为\[1, 128\)，不能和group\_ep相同。

-   **tp\_world\_size**（int）：可选参数，TP通信域size。有TP域通信才需要传参。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当有TP域通信时，取值范围\[0, 2\]，0和1表示无TP域通信，2表示有TP域通信。

-   **tp\_rank\_id**（int）：可选参数，TP通信域本卡ID。有TP域通信才需要传参。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当有TP域通信时，取值范围\[0, 1\]，同一个TP通信域中各卡的tp\_rank\_id不重复。无TP域通信时，传0即可。

-   **expert\_shard\_type**（int）：可选参数，表示共享专家卡排布类型。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当前仅支持0，表示共享专家卡排在MoE专家卡前面。

-   **shared\_expert\_num**（int）：可选参数，表示共享专家数量，一个共享专家可以复制部署到多个卡上。**预留参数，暂未使用，仅支持默认值0。**
-   **shared\_expert\_rank\_num**（int）：可选参数，表示共享专家卡数量。**预留参数，暂未使用，仅支持默认值0。**
-   **global\_bs**（int）：可选参数，表示EP域全局的batch size大小。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当每个rank的BS不同时，支持传入max\_bs\*ep\_world\_size，其中max\_bs表示单rank BS最大值；当每个rank的BS相同时，支持取值0或BS\*ep\_world\_size。

-   **out\_dtype**（int）：**预留参数，暂未使用，使用默认值即可**。
-   **comm\_quant\_mode**（int）：表示通信量化类型。**预留参数，暂未使用，使用默认值即可**。
-   **group\_list\_type**（int）：**预留参数，暂未使用，使用默认值即可**。
-   **norm\_eps**（float）：可选参数，用于防止add\_rms\_norm除0错误，默认值为1e-6。

## 返回值说明<a name="zh-cn_topic_0000002322738573_section1370204314220"></a>

-   **y**（Tensor）：表示combine处理后的token进行add\_rms\_norm计算后的结果，要求是3D的Tensor，shape为\(BS, 1, H\)，数据类型与输入residual\_x保持一致，数据格式为ND，不支持非连续的Tensor。
-   **rstd\_out**（Tensor）：表示add\_rms\_norm的输出结果，要求是3D的Tensor，shape为\(BS, 1, 1\)，数据类型支持float，数据格式为ND，不支持非连续的Tensor。
-   **x**（Tensor）：表示combine处理后的token进行add计算后的结果，要求是3D的Tensor，shape为\(BS, 1, H\)，数据类型与输入residual\_x保持一致，数据格式为ND，不支持非连续的Tensor。

## 约束说明<a name="zh-cn_topic_0000002322738573_section470214314214"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。
-   参数里Shape使用的变量如下：
    -   A：表示本卡发送的最大token数量，取值范围如下
        -   对于共享专家，要满足A=global\_bs\*shared\_expert\_num/shared\_expert\_rank\_num。
        -   对于MoE专家，当global\_bs为0时，要满足A\>=BS\*ep\_world\_size\*min\(local\_expert\_num, K\)；当global\_bs非0时，要满足A\>=global\_bs\* min\(local\_expert\_num, K\)。

    -   H：表示hidden size隐藏层大小。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围\[1024, 7168\]，且保证是32的整数倍。

    -   BS：表示待发送的token数量。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围为0<BS≤512。

    -   K：表示选取topK个专家，取值范围为0<K≤8同时满足0<K≤moe\_expert\_num。
    -   local\_expert\_num：表示本卡专家数量。
        -   对于共享专家卡，local\_expert\_num=1
        -   对于MoE专家卡，local\_expert\_num=moe\_expert\_num/\(ep\_world\_size-shared\_expert\_rank\_num\)，当local\_expert\_num\>1时，不支持TP域通信。

## 调用示例<a name="zh-cn_topic_0000002322738573_section9702174311218"></a>

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
    sharedExpertRankNum = 0                      # 共享专家数
    moeExpertNum = 32                            # moe专家数
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
    sharedExpertRankNum = 0                      # 共享专家数
    moeExpertNum = 32                            # moe专家数
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
    
        # 创建输入tensor
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
                               rank // tp_world_size,rank % tp_world_size, 0, sharedExpertRankNum, moeExpertNum, scales,
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

