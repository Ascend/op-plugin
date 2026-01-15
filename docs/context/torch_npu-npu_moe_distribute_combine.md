# torch\_npu.npu\_moe\_distribute\_combine<a name="ZH-CN_TOPIC_0000002309174912"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>  | √   |

## 功能说明<a name="zh-cn_topic_0000002168254826_section14441124184110"></a>

-   API功能：先进行reduce\_scatterv通信，再进行alltoallv通信，最后将接收的数据整合（乘权重再相加）。需与[torch\_npu.npu\_moe\_distribute\_dispatch](torch_npu-npu_moe_distribute_dispatch.md)配套使用，相当于按npu\_moe\_distribute\_dispatch算子收集数据的路径原路返回。

-   计算公式：

    $$
    rs\_out = ReduceScatterV(expend\_x)\\
    ata\_out = AlltoAllv(rs\_out)\\
    x = Sum(expert\_scales * ata\_out + expert\_scales * shared\_expert\_x)

    $$

## 函数原型<a name="zh-cn_topic_0000002168254826_section45077510411"></a>

```
torch_npu.npu_moe_distribute_combine(expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num, *, tp_send_counts=None, x_active_mask=None, activation_scale=None, weight_scale=None, group_list=None, expand_scales=None, shared_expert_x=None, group_tp="", tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, global_bs=0, out_dtype=0, comm_quant_mode=0, group_list_type=0) -> Tensor
```

## 参数说明<a name="zh-cn_topic_0000002168254826_section112637109429"></a>

-   **expand\_x** (`Tensor`)：必选参数。根据`expert_ids`进行扩展过的token特征，要求为2维张量，shape为\(max\(tp\_world\_size, 1\) \*A, H\)，数据类型支持`bfloat16`、`float16`，数据格式为$ND$，支持非连续的Tensor。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持共享专家场景。

-   **expert\_ids** (`Tensor`)：必选参数。每个token的topK个专家索引，要求为2维张量，shape为\(BS, K\)。数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_dispatch](torch_npu-npu_moe_distribute_dispatch.md)的`expert_ids`输入，张量里value取值范围为\[0, moe\_expert\_num\)，且同一行中的K个value不能重复。
-   **expand\_idx** (`Tensor`)：必选参数。表示给同一专家发送的token个数，要求为1维张量。数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_dispatch](torch_npu-npu_moe_distribute_dispatch.md)的`expand_idx`输出。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：要求shape为\(BS \* K,\)。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求shape为\(BS \* K,\)。

-   **ep\_send\_counts** (`Tensor`)：必选参数。示本卡每个专家发给EP（Expert Parallelism）域每个卡的token数（token数以前缀和的形式表示），要求为1维张量。数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_dispatch](torch_npu-npu_moe_distribute_dispatch.md)的`ep_recv_counts`输出。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：要求shape为\(moe\_expert\_num+2\*global\_bs\*K\*server\_num,\)，前`moe_expert_num`个数表示在EP通信域内，该卡上每个专家收到来自其他各卡的token数（以前缀和的形式表示），2\*global\_bs\*K\*server\_num用于存储机间和机内通信前，combine可提前做reduce操作的token个数和通信区偏移量，`global_bs`传入0时此处按照bs\*ep\_world\_size计算。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求shape为\(ep\_world\_size\*max\(tp\_world\_size, 1\)\*local\_expert\_num,\)。

-   **expert\_scales** (`Tensor`)：必选参数。表示每个token的topK个专家的权重，要求为2维张量，shape为\(BS, K\)，其中共享专家不需要乘权重系数，直接相加即可。数据类型支持`float`，数据格式为$ND$，支持非连续的Tensor。
-   **group\_ep** (`str`)：必选参数。EP通信域名称，专家并行的通信域。字符串长度范围为\[1, 128\)。<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>时不能和`group_tp`相同。
-   **ep\_world\_size** (`int`)：必选参数，EP通信域size。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：取值支持16、32、64。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值支持8、16、32、64、128、144、256、288。

-   **ep\_rank\_id** (`int`)：必选参数，EP通信域本卡ID，取值范围\[0, ep\_world\_size\)，同一个EP通信域中各卡的`ep_rank_id`不重复。
-   **moe\_expert\_num** (`int`)：必选参数，MoE专家数量，取值范围\[1, 512\]，并且满足以下条件：moe\_expert\_num\%\(ep\_world\_size - shared\_expert\_rank\_num\)\=0。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：还需满足moe\_expert\_num\/\(ep\_world\_size - shared\_expert\_rank\_num\) <= 24。
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
-   **tp\_send\_counts** (`Tensor`)：可选参数，表示本卡每个专家发给TP（Tensor Parallelism）通信域每个卡的数据量。对应[torch\_npu.npu\_moe\_distribute\_dispatch](torch_npu-npu_moe_distribute_dispatch.md)的`tp_recv_counts`输出。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持TP通信域，使用默认输入None。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持TP通信域，要求为一个1维张量，shape为\(tp\_world\_size,\)，数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。

-   **x\_active\_mask** (`Tensor`)：预留参数，暂未使用，使用默认值即可。
-   **activation\_scale** (`Tensor`)：预留参数，暂未使用，使用默认值即可。
-   **weight\_scale** (`Tensor`)：预留参数，暂未使用，使用默认值即可。
-   **group\_list** (`Tensor`)：预留参数，暂未使用，使用默认值即可。
-   **expand\_scales** (`Tensor`)：对应[torch\_npu.npu\_moe\_distribute\_dispatch](torch_npu-npu_moe_distribute_dispatch.md)的`expand_scales`输出。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：必选参数，要求为1维张量，shape为\(A,\)，数据类型支持`float`，数据格式为$ND$，支持非连续的Tensor。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：暂不支持该参数，使用默认值即可。

-   **shared\_expert\_x** (`Tensor`)：预留参数，暂未使用，使用默认值即可。

-   **group\_tp** (`str`)：可选参数，TP通信域名称，数据并行的通信域。有TP域通信才需要传参，若无TP域通信，使用默认值""即可。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：eager模式使用默认值即可，图模式传入与`group_ep`相同。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：字符串长度范围为\[1, 128\)，不能和`group_ep`相同。

-   **tp\_world\_size** (`int`)：可选参数，TP通信域size。有TP域通信才需要传参。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持TP域通信，使用默认值0即可。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当有TP域通信时，取值范围\[0, 2\]，0和1表示无TP域通信，2表示有TP域通信。

-   **tp\_rank\_id** (`int`)：可选参数，TP通信域本卡ID。有TP域通信才需要传参。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持TP域通信，使用默认值0即可。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当有TP域通信时，取值范围\[0, 1\]，同一个TP通信域中各卡的`tp_rank_id`不重复。无TP域通信时，传0即可。

-   **expert\_shard\_type** (`int`)：表示共享专家卡排布类型。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：暂不支持该参数，使用默认值即可。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当前仅支持0，表示共享专家卡排在MoE专家卡前面。

-   **shared\_expert\_num** (`int`)：表示共享专家数量，一个共享专家可以复制部署到多个卡上。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：暂不支持该参数，使用默认值即可。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅支持1，默认值为1。

-   **shared\_expert\_rank\_num** (`int`)：可选参数，表示共享专家卡数量。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持共享专家，传0即可。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围\[0, ep\_world\_size\)。取0表示无共享专家，不取0时需满足ep\_world\_size%shared\_expert\_rank\_num=0。

-   **global\_bs** (`int`)：可选参数，表示EP域全局的BS(batch size)大小。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：当每个rank的BS不同时，支持传入max\_bs\*ep\_world\_size或者256\*ep\_world\_size，其中max\_bs表示单rank BS最大值，建议按max\_bs\*ep\_world\_size传入，固定按256\*ep\_world\_size传入，在后续版本BS大于256的场景下会无法支持；当每个rank的BS相同时，支持取值0或BS\*ep\_world\_size。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当每个rank的BS不同时，支持传入max\_bs\*ep\_world\_size，其中max\_bs表示单rank BS最大值；当每个rank的BS相同时，支持取值0或BS\*ep\_world\_size。

-   **out\_dtype** (`int`)：预留参数，暂未使用，使用默认值即可。
-   **comm\_quant\_mode** (`int`)：表示通信量化类型。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持取0和2。0表示通信时不量化，2表示通信时进行`int8`量化。仅当HCCL\_INTRA\_PCIE\_ENABLE=1且HCCL\_INTRA\_ROCE\_ENABLE=0且驱动版本不低于25.0.RC1.1时才支持取2。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持取0和2。0表示通信时不量化，2表示通信时进行`int8`量化。当且仅当`tp_world_size`不等于2时，可以使能`int8`量化。

-   **group\_list\_type** (`int`)：预留参数，暂未使用，使用默认值即可。

## 返回值说明<a name="zh-cn_topic_0000002168254826_section22231435517"></a>
`Tensor`

表示处理后的token，要求为2维张量，shape为\(BS, H\)，数据类型支持`bfloat16`、`float16`，类型与输入`expand_x`保持一致，数据格式为$ND$，不支持非连续的Tensor。

## 约束说明<a name="zh-cn_topic_0000002168254826_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持静态图模式，`npu_moe_distribute_dispatch`和`npu_moe_distribute_combine`必须配套使用。
-   在不同产品型号、不同通信算法或不同版本中，`npu_moe_distribute_dispatch`的Tensor输出`expand_idx`、`ep_recv_counts`、`tp_recv_counts`、`expand_scales`中的元素值可能不同，使用时直接将上述Tensor传给`npu_moe_distribute_combine`对应参数即可，模型其他业务逻辑不应对其存在依赖。
-   调用接口过程中使用的`group_ep`、`ep_world_size`、`moe_expert_num`、`group_tp`、`tp_world_size`、`expert_shard_type`、`shared_expert_num`、`shared_expert_rank_num`、`global_bs`参数取值所有卡需保持一致，`group_ep`、`ep_world_size`、`moe_expert_num`、`group_tp、tp_world_size`、`expert_shard_type`、`global_bs`网络中不同层中也需保持一致，且和[torch\_npu.npu\_moe\_distribute\_dispatch](torch_npu-npu_moe_distribute_dispatch.md)对应参数也保持一致。
-   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
-   参数里Shape使用的变量如下：
    -   A：表示本卡接收的最大token数量，取值范围如下：
        -   对于共享专家，当`global_bs`为0时，要满足A=BS\*shared\_expert\_num/shared\_expert\_rank\_num；当`global_bs`非0时，要满足A=global\_bs\*shared\_expert\_num/shared\_expert\_rank\_num。
        -   对于MoE专家，当`global_bs`为0时，要满足A\>=BS\*ep\_world\_size\*min\(local\_expert\_num, K\)；当`global_bs`非0时，要满足A\>=global\_bs\* min\(local\_expert\_num, K\)。

    -   H：表示hidden size隐藏层大小。
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：取值范围\(0, 7168\]，且保证是32的整数倍。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅支持 7168。

    -   BS：表示待发送的token数量。
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：取值范围为0<BS≤256。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围为0<BS≤512。

    -   K：表示选取topK个专家，需满足0<K≤moe\_expert\_num。
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：保证取值范围为0<K≤16。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：保证取值范围为0<K≤8。

    -   server\_num：表示服务器的节点数，取值只支持2、4、8。
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：仅该场景的shape使用了该变量。

    -   local\_expert\_num：表示本卡专家数量。
        -   对于共享专家卡，local\_expert\_num=1
        -   对于MoE专家卡，local\_expert\_num=moe\_expert\_num/\(ep\_world\_size-shared\_expert\_rank\_num)，当local\_expert\_num\>1时，不支持TP域通信。

-   HCCL通信域缓存区大小:

    调用本接口前需检查通信域缓存区大小取值是否合理，单位MB，不配置时默认为200MB。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
        该场景支持通过环境变量HCCL\_BUFFSIZE配置。
        - 设置大小要求\>=2\*\(BS\*ep\_world\_size\*min\(local\_expert\_num, K\)\*H\*sizeof\(uint16\)+2MB\)。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
        该场景不仅支持通过环境变量HCCL\_BUFFSIZE配置，还支持通过hccl_buffer_size配置（参考《[PyTorch训练模型迁移调优](https://hiascend.com/document/redirect/canncommercial-ptmigr)》中“性能调优>性能调优方法>通信优化>优化方法>hccl_buffer_size”章节）。
        - ep通信域内：设置大小要求\>=2且满足1024\^2\*\(HCCL\_BUFFSIZE\-2\)\/2\>=BS\*2\*\(H\+128\)\*\(ep\_world\_size\*local\_expert\_num\+K\+1\)，local\_expert\_num需使用MoE专家卡的本卡专家数。
        - tp通信域内：设置大小要求\>=A * (H * 2 + 128) * 2。
-   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：配置环境变量HCCL\_INTRA\_PCIE\_ENABLE=1和HCCL\_INTRA\_ROCE\_ENABLE=0可以减少跨机通信数据量，提升算子性能。此时要求HCCL\_BUFFSIZE\>=moe\_expert\_num\*BS\*\(H\*sizeof\(dtype_x\)+4\*\(\(K+7\)/8\*8\)\*sizeof\(uint32\)\)+4MB+100MB。并且，对于入参moe\_expert\_num，只要求moe\_expert\_num\%\(ep\_world\_size - shared\_expert\_rank\_num\)\=0，不要求moe\_expert\_num\/\(ep\_world\_size - shared\_expert\_rank\_num\) <= 24。

-   本文公式中的“/”表示整除。

-   通信域使用约束：

    -   一个模型中的`npu_moe_distribute_dispatch`和`npu_moe_distribute_combine`算子仅支持相同EP通信域，且该通信域中不允许有其他算子。

    -   一个模型中的`npu_moe_distribute_dispatch`和`npu_moe_distribute_combine`算子仅支持相同TP通信域或都不支持TP通信域，有TP通信域时该通信域中不允许有其他算子。

    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：一个通信域内的节点需在一个超节点内，不支持跨超节点。

-   组网约束：
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：多机场景仅支持交换机组网，不支持双机直连组网。

## 调用示例<a name="zh-cn_topic_0000002168254826_section14459801435"></a>

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

        model = MOE_DISTRIBUTE_GRAPH_Model()
        model = model.npu()
        npu_backend = torchair.get_npu_backend()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        output = model.forward(x, expert_ids, ep_hcomm_info, tp_hcomm_info, ep_world_size, tp_world_size,
                               rank // tp_world_size,rank % tp_world_size, 0, sharedExpertRankNum, moeExpertNum, scales,
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

