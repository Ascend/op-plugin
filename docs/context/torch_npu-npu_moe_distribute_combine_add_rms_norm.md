# torch\_npu.npu\_moe\_distribute\_combine\_add\_rms\_norm<a name="ZH-CN_TOPIC_0000002384325441"></a>

## 产品支持情况<a name="zh-cn_topic_0000002322738573_section1369303644412"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √   |

## 功能说明<a name="zh-cn_topic_0000002322738573_section1470016430218"></a>

-   API功能：

    需与[torch_npu.npu_moe_distribute_dispatch_v2](torch_npu-npu_moe_distribute_dispatch_v2.md)配套使用，相当于按`npu_moe_distribute_dispatch_v2`算子收集数据的路径原路返回后对数据进行`add_rms_norm`操作。
     - 支持数据整合功能，即对moe_distribute_combine、add及rms_norm进行功能融合；
     - 支持动态缩容场景，在创建通信域后，出现故障卡，将故障卡从通信域中剔除，算子可正常执行，无需重新编译；
     - 支持特殊专家场景。
-   计算公式：
    - 数据整合功能：

        $$
        rsOut = ReduceScatterV(expand\_x)\\
        ataOut = AllToAllV(rs\_out)\\
        combineOut = Sum(expert\_scales * ata\_out + expert\_scales * shared\_expert\_x)\\
        x = combine\_out + residual\_x\\
        y = \frac{x}{RMS(x)} * gamma,\quad\text{where}RMS(x) = \sqrt{\frac{1}{H}\sum_{i=1}^{H}x_{i}^{2}+norm\_eps}\\
        $$

    -   特殊专家场景：

        - 零专家场景（zero_expert_num ≠ 0）：

            $$Moe(ori\_x)=0$$

        - 拷贝专家场景（copy_expert_num ≠ 0）：

            $$Moe(ori\_x)=ori\_x$$
      
        - 常量专家场景（const_expert_num ≠ 0）：

            $$Moe(ori\_x)=const\_expert\_alpha\_1*ori\_x+const\_expert\_alpha\_2*const\_expert\_v$$

## 函数原型<a name="zh-cn_topic_0000002322738573_section470115437220"></a>

```
torch_npu.npu_moe_distribute_combine_add_rms_norm(expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, residual_x, gamma, group_ep, ep_world_size, ep_rank_id, moe_expert_num, *, tp_send_counts=None, x_active_mask=None, activation_scale=None, weight_scale=None, group_list=None, expand_scales=None, shared_expert_x=None, Tensor elastic_info=None, Tensor ori_x=None, Tensor const_expert_alpha_1=None, Tensor const_expert_alpha_2=None, Tensor const_expert_v=None, group_tp="", tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, global_bs=0, out_dtype=0, comm_quant_mode=0, group_list_type=0, norm_eps=1e-06, int zero_expert_num=0, int copy_expert_num=0, int const_expert_num=0) -> (Tensor, Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002322738573_section187018431529"></a>

-   **expand\_x**（`Tensor`）：必选参数，根据`expert_ids`进行扩展过的token特征，要求为2D的Tensor，shape为\(max\(`tp_world_size`, 1\) \*A, H\)，数据类型支持`bfloat16`，数据格式为ND，支持非连续的Tensor。
-   **expert\_ids**（`Tensor`）：必选参数，每个token的topK个专家索引，要求为2D的Tensor，shape为\(BS, K\)。数据类型支持`int32`，数据格式为ND，支持非连续的Tensor。对应`torch_npu.npu_moe_distribute_dispatch`的`expert_ids`输入，张量里value取值范围为\[0, `moe_expert_num`\)，且同一行中的K个value不能重复。
-   **expand\_idx**（`Tensor`）：必选参数，表示给同一专家发送的token个数，要求是1D的Tensor，shape为\(A\*128, \)。数据类型支持int32，数据格式为ND，支持非连续的Tensor。对应`torch_npu.npu_moe_distribute_dispatch`的`expand_idx`输出。
-   **ep\_send\_counts**（`Tensor`）：必选参数，表示本卡每个专家发给EP（Expert Parallelism）域每个卡的数据量，要求是1D的Tensor 。数据类型支持`int32`，数据格式为ND，支持非连续的Tensor。对应`torch_npu.npu_moe_distribute_dispatch`的`ep_recv_counts`输出。
    <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求shape为\(`ep_world_size`\*max\(`tp_world_size`, 1\)\*local\_expert\_num, \)。

-   **expert\_scales**（`Tensor`）：必选参数，表示每个token的topK个专家的权重，要求是2D的Tensor，shape为\(BS, K\)，其中共享专家不需要乘权重系数，直接相加即可。数据类型支持`float`，数据格式为ND，支持非连续的Tensor。
-   **residual\_x**（`Tensor`）：必选参数，表示处理后的token需要add的参数，要求是3D的Tensor，shape为\(BS, 1, H\)。数据类型支持`bfloat16`，数据格式为ND，支持非连续的Tensor。
-   **gamma**（`Tensor`）：必选参数，表示rms\_norm的权重，要求是1D的Tensor，shape为\(H, \)。数据类型支持`bfloat16`，数据格式为ND，支持非连续的Tensor。
-   **group\_ep**（`str`）：必选参数，EP通信域名称，专家并行的通信域。字符串长度范围为\[1, 128\)，不能和`group_tp`相同。
-   **ep\_world\_size**（`int`）：必选参数，EP通信域size。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值支持\[2, 768\]。

-   **ep\_rank\_id**（`int`）：必选参数，EP通信域本卡ID，取值范围\[0, `ep_world_size`\)，同一个EP通信域中各卡的ep\_rank\_id不重复。
-   **moe\_expert\_num**（`int`）：必选参数，MoE专家数量，取值范围\[1, 1024\]，并且满足`moe_expert_num`%\(`ep_world_size`-`shared_expert_rank_num`\)=0。
-   **tp\_send\_counts**（`Tensor`）：可选参数，表示本卡每个专家发给TP（Tensor  Parallelism）通信域每个卡的数据量。对应`torch_npu.npu_moe_distribute_dispatch`的`tp_recv_counts`输出。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持TP通信域，要求是一个1D Tensor，shape为\(`tp_world_size`, \)，数据类型支持`int32`，数据格式要求为ND，支持非连续的Tensor。

-   **x\_active\_mask**（`Tensor`）：Tensor类型，
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求是一个1D或者2D Tensor。当输入为1D时，shape为\(BS, \); 当输入为2D时，shape为\(BS, K\)。数据类型支持bool，数据格式要求为ND，支持非连续的Tensor。当输入为1D时，参数为true表示对应的token参与通信，true必须排到false之前，例：{true, false, true} 为非法输入；当输入为2D时，参数为true表示当前token对应的`expert_ids`参与通信，若当前token对应的K个`bool`值全为false，表示当前token不会参与通信。默认所有token都会参与通信。当每张卡的BS数量不一致时，所有token必须全部有效。

-   **activation\_scale**（`Tensor`）：可选参数，**预留参数暂未使用，使用默认值即可。**
-   **weight\_scale**（`Tensor`）：可选参数，**预留参数暂未使用，使用默认值即可。**
-   **group\_list**（`Tensor`）：可选参数，**预留参数暂未使用，使用默认值即可。**
-   **expand\_scales**（`Tensor`）：可选参数，对应`torch_npu.npu_moe_distribute_dispatch`的`expand_scales`输出。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：暂不支持该参数，使用默认值即可。

-   **shared\_expert\_x**（`Tensor`）：可选参数，数据类型需与`expand_x`保持一致。仅在共享专家卡数量`shared_expert_rank_num`为0的场景下使用，表示共享专家token，在combine时需要加上。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型需与`expand_x`保持一致，shape为\[BS, H\]。

-   **elastic\_info** (`Tensor`)：可选参数，表示EP通信域的动态缩容信息。当某些通信卡因异常从通信域中被剔除，实际参与通信的卡数与创建通信域时不一致，可从本参数中获取当前部署信息。可选择传入有效数据或填空指针，传入空指针时表示不使能动态缩容功能；当传入有效数据时，要求是一个1D的Tensor，shape为(4 + 2\*ep\_world\_size,)，数据类型支持int32；数据格式要求为ND，支持非连续的Tensor。Tensor中的前四个数字分别表示（是否缩容，缩容后实际rnak数，缩容后共享专家使用rank数，缩容后moe专家个数），后2\*ep\_world\_size表示两个rank映射表，第一个Table的映射关系为Table1[ep\_rank\_id]=local\_ep\_rank\_id或-1，-1表示epRankId这张卡从通信域中剔除，localEpRankId表示缩容后本卡在新EP通信域中的rankIndex，第二个Table映射关系为Table2[local\_ep\_rank\_id]=ep\_rank\_id。</term>

-   **ori\_x** (`Tensor`)：可选参数，表示未经过FFN的token数据，在使能copy_expert或使能const_expert的场景下需要本输入数据。可选择传入有效数据或填空指针，当copy_expert_num不为零或const_expert_num不为零时必须传入有效输入；当传入有效数据时，要求是一个2D的Tensor，shape为(BS,H)，数据类型需跟expand_x保持一致；数据格式要求为ND，支持非连续的Tensor。

-   **const\_expert\_alpha\_1** (`Tensor`)：可选参数，在使能const_expert的场景下需要输入的计算系数。可选择传入有效数据或填空指针，当const_expert_num不为零时必须传入有效输入；当传入有效数据时，要求是一个1D的Tensor，shape为(const_expert_num,)，数据类型需跟expand_x保持一致；数据格式要求为ND，支持非连续的Tensor。

-   **const\_expert\_alpha\_2** (`Tensor`)：可选参数，在使能const_expert的场景下需要输入的计算系数。可选择传入有效数据或填空指针，当const_expert_num不为零时必须传入有效输入；当传入有效数据时，要求是一个1D的Tensor，shape为(const_expert_num,)，数据类型需跟expand_x保持一致；数据格式要求为ND，支持非连续的Tensor。

-   **const\_expert\_v** (`Tensor`)：可选参数，在使能const_expert的场景下需要输入的计算系数。可选择传入有效数据或填空指针，当const_expert_num不为零时必须传入有效输入；当传入有效数据时，要求是一个2D的Tensor，shape为(const_expert_num,H)，数据类型需跟expand_x保持一致；数据格式要求为ND，支持非连续的Tensor。

-   **group\_tp**（`str`）：可选参数，TP通信域名称，数据并行的通信域。有TP域通信才需要传参。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当有TP域通信时，字符串长度范围为\[1, 128\)，不能和`group_ep`相同。

-   **tp\_world\_size**（`int`）：可选参数，TP通信域size。有TP域通信才需要传参。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当有TP域通信时，取值范围\[0, 2\]，0和1表示无TP域通信，2表示有TP域通信。

-   **tp\_rank\_id**（`int`）：可选参数，TP通信域本卡ID。有TP域通信才需要传参。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当有TP域通信时，取值范围\[0, 1\]，同一个TP通信域中各卡的tp\_rank\_id不重复。无TP域通信时，传0即可。

-   **expert\_shard\_type**（`int`）：可选参数，表示共享专家卡排布类型。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当前仅支持0，表示共享专家卡排在MoE专家卡前面。

-   **shared\_expert\_num**（`int`）：可选参数，表示共享专家数量，一个共享专家可以复制部署到多个卡上。**预留参数暂未使用，仅支持默认值0。**
-   **shared\_expert\_rank\_num**（`int`）：可选参数，表示共享专家卡数量。**预留参数暂未使用，仅支持默认值0。**
-   **global\_bs**（`int`）：可选参数，表示EP域全局的batch size大小。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当每个rank的BS不同时，支持传入max\_bs\*`ep_world_size`，其中max\_bs表示单rank BS最大值；当每个rank的BS相同时，支持取值0或BS\*`ep_world_size`。

-   **out\_dtype**（`int`）：可选参数，**预留参数暂未使用，使用默认值即可**。
-   **comm\_quant\_mode**（`int`）：可选参数，表示通信量化类型。**预留参数暂未使用，使用默认值即可**。
-   **group\_list\_type**（`int`）：可选参数，**预留参数暂未使用，使用默认值即可**。
-   **norm\_eps**（`float`）：可选参数，用于防止add\_rms\_norm除0错误，默认值为1e-6。

-   **zero\_expert\_num** (`int`)：可选参数，表示零专家的数量。取值范围\[0, MAX_INT32\)，其中MAX_INT32值为2147483647，合法的零专家的ID值是\[moe\_expert\_num, moe\_expert\_num+zero\_expert\_num\)。

-   **copy\_expert\_num** (`int`)：可选参数，表示copy专家的数量。取值范围\[0, MAX_INT32\)，其中MAX_INT32值为2147483647，合法的零专家的ID值是\[moe\_expert\_num, moe\_expert\_num+zero\_expert\_num+copy\_expert\_num\)。

-   **const\_expert\_num** (`int`)：可选参数，表示常量专家的数量。取值范围\[0, MAX_INT32\)，其中MAX_INT32值为2147483647，合法的零专家的ID值是\[moe\_expert\_num, moe\_expert\_num+zero\_expert\_num+copy\_expert\_num+const\_expert\_num\)。

## 返回值说明<a name="zh-cn_topic_0000002322738573_section1370204314220"></a>

-   **y**（`Tensor`）：表示combine处理后的token进行add\_rms\_norm计算后的结果，要求是3D的Tensor，shape为\(BS, 1, H\)，数据类型与输入`residual_x`保持一致，数据格式为ND，不支持非连续的Tensor。
-   **rstd\_out**（`Tensor`）：表示add\_rms\_norm的输出结果，要求是3D的Tensor，shape为\(BS, 1, 1\)，数据类型支持`float`，数据格式为ND，不支持非连续的Tensor。
-   **x**（`Tensor`）：表示combine处理后的token进行add计算后的结果，要求是3D的Tensor，shape为\(BS, 1, H\)，数据类型与输入`residual_x`保持一致，数据格式为ND，不支持非连续的Tensor。

## 约束说明<a name="zh-cn_topic_0000002322738573_section470214314214"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   动态缩容后的部署信息通过elastic_info参数传递给算子，无需修改其他参数。动态缩容后，MOE专家卡上的本卡部署MOE专家数需与缩容前保持一致。
- 动态缩容功能不支持在TP并行场景下使能。
- 调用接口过程中使用的expert_ids、x_active_mask、elastic_info、group_ep、ep_world_size、moe_expert_num、group_tp、tp_world_size、expert_shard_type、shared_expert_num、shared_expert_rank_num、global_bs、comm_alg、zero_expert_num、copy_expert_num、const_expert_num参数、HCCL_BUFFSIZE取值所有卡需保持一致，网络中不同层中也需保持一致，且和[torch\_npu.npu\_moe\_distribute\_dispatch\_v2](torch_npu-npu_moe_distribute_dispatch_v2.md)对应参数也保持一致。
-   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
-   moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num < MAX_INT32，其中MAX_INT32值为2147483647。
-   参数里Shape使用的变量如下：
    - A：表示本卡需要分发的最大token数量，取值范围如下：
        - 当globalBs为0时，要满足A >= Bs * epWorldSize * min(localExpertNum, K)；
        - 当globalBs非0时，要满足A >= globalBs * min(localExpertNum, K)。   

    -   H：表示hidden size隐藏层大小。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围\[1024, 8192\]。

    -   BS：表示待发送的token数量。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围为0<BS≤512。

    -   K：表示选取topK个专家，取值范围为0<K≤8同时满足0 < K ≤ moe\_expert\_num + zero_expert_num + copy_expert_num + const_expert_num。
    -   local\_expert\_num：表示本卡专家数量。
        -   对于共享专家卡，local\_expert\_num=1
        -   对于MoE专家卡，local\_expert\_num=moe\_expert\_num/\(ep\_world\_size-shared\_expert\_rank\_num\)，当local\_expert\_num\>1时，不支持TP域通信。

-   HCCL_BUFFSIZE:
    调用本接口前需检查HCCL\_BUFFSIZE环境变量取值是否合理该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。要求 \>= 2且满足\>= 2 \* \(local\_expert\_num \* max\_bs \* ep\_world\_size \* Align512\(Align32\(2 \* H\) + 64\) + \(K + shared\_expert\_num\) \* max\_bs \* Align512\(2 \* H\)\)，local\_expert\_num表示需使用MoE专家卡的本卡专家数。

-   通信域使用约束：

    -   一个模型中的npu\_moe\_distribute\_dispatch\_v2和npu\_moe\_distribute\_combine\_add\_rms\_norm算子仅支持相同EP通信域，且该通信域中不允许有其他算子。

    -   一个模型中的npu\_moe\_distribute\_dispatch\_v2和npu\_moe\_distribute\_combine\_add\_rms\_norm算子仅支持相同TP通信域或都不支持TP通信域，有TP通信域时该通信域中不允许有其他算子。


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
    import time
    
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
    shared_expert_rank_num = 0                      # 共享专家数
    moe_expert_num = 32                            # moe专家数
    bs = 8                                       # token数量
    h = 7168                                     # 每个token的长度
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
        const_expert_alpha_1 = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1

    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
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
    
        # 创建输入tensor
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = torch.tensor([[ 5,  7, 17,  4,  2,  6, 11, 16],
            [10, 12, 13, 15, 19,  4, 18,  1],
            [19, 33,  1, 17,  9,  5,  0, 32],
            [19, 11, 17,  0, 10,  5,  7,  9],
            [10, 16, 11, 17, 33,  8,  9,  3],
            [12, 19,  5,  7,  1,  3, 18, 16],
            [11,  9, 13, 16, 12, 33, 17, 14],
            [16,  4,  9,  5,  0, 10, 11, 17]], dtype=torch.int32).npu()

        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + moe_expert_num, h) if shared_expert_rank_num else (moe_expert_num, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None

        elastic_info = torch.tensor([ 1, 10, 0, 20,
                                     -1,  0, 1,  2, -1, 3, -1,  4, -1,  5,  6,  7, -1,  8,  9, -1,
                                      1,  2, 3,  5,  7, 9, 10, 11, 13, 14, -1, -1, -1, -1, -1,  -1], dtype=torch.int32).npu()
        available_ranks = [1, 2, 3, 5, 7, 9, 10, 11, 13, 14]
        const_expert_alpha_1 = gen_const_expert_alpha_1().npu()
        const_expert_alpha_2 = gen_const_expert_alpha_2().npu()
        const_expert_v = gen_const_expert_v().npu()

        out = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)

        if rank in available_ranks:
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
                elastic_info=elastic_info,
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
                elastic_info=elastic_info,
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
            print("shared_expert_rank_num 不能大于 ep_world_size")
            exit(0)
    
        if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
            print("ep_world_size 必须是 shared_expert_rank_num的整数倍")
            exit(0)
    
        if moe_expert_num % moe_rank_num != 0:
            print("moe_expert_num 必须是 moe_rank_num 的整数倍")
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
    import time
    
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
    shared_expert_rank_num = 0                      # 共享专家数
    moe_expert_num = 32                            # moe专家数
    bs = 8                                       # token数量
    h = 7168                                     # 每个token的长度
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
        const_expert_alpha_1 = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1

    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
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
    
        # 创建输入tensor
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

        elastic_info = torch.tensor([1, 10, 0, 20,
                                    -1, 0, 1, 2, -1, 3, -1, 4, -1, 5, 6, 7, -1, 8, 9, -1,
                                     1, 2, 3, 5, 7, 9, 10, 11, 13, 14, -1, -1, -1, -1, -1, -1],dtype=torch.int32).npu()
        available_ranks = [1, 2, 3, 5, 7, 9, 10, 11, 13, 14]

        const_expert_alpha_1 = gen_const_expert_alpha_1().npu()
        const_expert_alpha_2 = gen_const_expert_alpha_2().npu()
        const_expert_v = gen_const_expert_v().npu()
    
        out = warm_up_dispatch(rank, ep_hcomm_info, tp_hcomm_info)
        model = MOE_DISTRIBUTE_GRAPH_Model()
        model = model.npu()
        npu_backend = torchair.get_npu_backend()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        if rank in available_ranks:
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
            print("shared_expert_rank_num 不能大于 ep_world_size")
            exit(0)
    
        if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
            print("ep_world_size 必须是 shared_expert_rank_num的整数倍")
            exit(0)
    
        if moe_expert_num % moe_rank_num != 0:
            print("moe_expert_num 必须是 moe_rank_num 的整数倍")
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

