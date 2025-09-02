# torch\_npu.npu\_moe\_distribute\_dispatch\_v2<a name="ZH-CN_TOPIC_0000002343094193"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>  | √   |

## 功能说明<a name="zh-cn_topic_0000002203575833_section14441124184110"></a>

-   算子功能：需与[torch\_npu.npu\_moe\_distribute\_combine\_v2](torch_npu-npu_moe_distribute_combine_v2.md)配套使用，完成MoE的并行部署下的token dispatch\_v2与combine\_v2。对token数据先进行量化（可选），再进行EP（Expert Parallelism）域的alltoallv通信，再进行TP（Tensor  Parallelism）域的allgatherv通信（可选）。
-   支持动态缩容场景，支持在创建通信域后，出现故障卡，将故障卡从通信域中剔除，算子可正常执行，无需重新编译；
-   支持零计算专家场景————zeroExpert:Moe(x)=0, copyExpert:Moe(x)=x, constExpert:Moe(x)=alpha1\*x+alpha2\*v。
-   计算公式：
    -   若`quant_mode`不为`2`，即非动态量化场景：

        ![](./figures/zh-cn_formulaimage_0000002244554688.png)

    -   若`quant_mode`为`2`，即动态量化场景：

        ![](./figures/zh-cn_formulaimage_0000002244394892.png)

## 函数原型<a name="zh-cn_topic_0000002203575833_section45077510411"></a>

```
torch_npu.npu_moe_distribute_dispatch_v2(x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, *, scales=None, x_active_mask=None, expert_scales=None, elastic_info=None, group_tp="", tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, quant_mode=0, global_bs=0, expert_token_nums_type=1, comm_alg="", zero_expert_num=0, copy_expert_num=0, const_expert_num=0) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002203575833_section112637109429"></a>

-   **x** (`Tensor`)：必选参数，表示计算使用的token数据，需根据`expert_ids`来发送给其他卡。要求为2维张量，shape为\(BS, H\)，表示有BS个token，数据类型支持`bfloat16`、`float16`，数据格式为$ND$，支持非连续的Tensor。
-   **expert\_ids** (`Tensor`)：必选参数，表示每个token的topK个专家索引，决定每个token要发给哪些专家。要求为2维张量，shape为\(BS, K\)，数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_combine\_v2](torch_npu-npu_moe_distribute_combine_v2.md)的`expert_ids`输入，张量里value取值范围为\[0, moe\_expert\_num\)，且同一行中的K个value不能重复。
-   **group\_ep** (`str`)：必选参数，EP通信域名称，专家并行的通信域。字符串长度范围为\[1,128\)，不能和`group_tp`相同。
-   **ep\_world\_size**(`int`)：必选参数，EP通信域size。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：取值支持16、32、64。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值支持\[2, 768\]。

-   **ep\_rank\_id** (`int`)：必选参数，EP通信域本卡ID，取值范围\[0, ep\_world\_size\)，同一个EP通信域中各卡的`ep_rank_id`不重复。
-   **moe\_expert\_num** (`int`)：必选参数，MoE专家数量，取值范围\[1, 1024\]，并且满足以下条件：moe\_expert\_num\%\(ep\_world\_size - shared\_expert\_rank\_num\)\=0。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：还需满足moe\_expert\_num\/\(ep\_world\_size - shared\_expert\_rank\_num\) <= 24。
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
-   **scales** (`Tensor`)：可选参数，表示每个专家的权重，非量化场景不传，动态量化场景可传可不传。若传值要求为2维张量，如果有共享专家，shape为\(shared\_expert\_num+moe\_expert\_num, H\)，如果没有共享专家，shape为\(moe\_expert\_num, H\)，数据类型支持`float`，数据格式为$ND$，不支持非连续的Tensor。
-   **x\_active\_mask** (`Tensor`)：可选参数，表示token是否参与通信。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：预留参数，暂未使用，使用默认值即可。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求是一个1维或者2维张量。当输入为1维时，shape为\(BS, \); 当输入为2维时，shape为\(BS, K\)。数据类型支持`bool`，数据格式要求为$ND$，支持非连续的Tensor。当输入为1维时，参数为true表示对应的token参与通信，true必须排到false之前，例：{true, false, true} 为非法输入；当输入为2D时，参数为true表示当前token对应的`expert_ids`参与通信，若当前token对应的K个`bool`值全为false，表示当前token不会参与通信。默认所有token都会参与通信。当每张卡的BS数量不一致时，所有token必须全部有效。

-   **expert\_scales** (`Tensor`)：可选参数，表示每个token的topK个专家权重。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：要求为2维张量，shape为\(BS, K\)，数据类型支持`float`，数据格式为$ND$，支持非连续的Tensor。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：暂不支持该参数，使用默认值即可。

-   **elastic\_info** (`Tensor`)：可选参数，表示EP通信域的动态缩容信息。当某些通信卡因异常被从通信域中剔除，实际参与通信的卡数与创建通信域时不一致，可从本参数中获取当前部署信息。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：预留参数，当前版本不支持，传空指针即可。</term>
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品：可选择传入有效数据或填空指针，传入空指针时表示不使能动态缩容功能；当传入有效数据时，要求是一个1D的Tensor，shape为(4 + 2\*ep\_world\_size,)，数据类型支持int32；数据格式要求为ND，支持非连续的Tensor。Tensor中的前四个数字分别表示(是否缩容，缩容后实际rnak数，共享专家使用rank数，moe专家个数)，后2*ep\_world\_size表示两个rank映射表，缩容后本卡在新EP通信域中的rank index使用local\_ep\_rank\_id表示，第一个Table的映射关系为Table1[ep\_rank\_id]=local\_ep\_rank\_id或-1，-1表示ep\_rank\_id这张卡从通信域中被剔除，第二个Table映射关系为Table2[local\_ep\_rank\_id]=ep\_rank\_id。</term>

-   **group\_tp** (`string`)：可选参数，TP通信域名称，数据并行的通信域。若有TP域通信需要传参，若无TP域通信，使用默认值即可。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：不支持TP域通信，使用默认值即可。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：字符串长度范围为\[1, 128\)，不能和`group_ep`相同。

-   **tp\_world\_size** (`int`)：可选参数，TP通信域size。有TP域通信才需要传参。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：不支持TP域通信，使用默认值0即可。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当有TP域通信时，取值范围\[0, 2\]，0和1表示无TP域通信，2表示有TP域通信。

-   **tp\_rank\_id** (`int`)：可选参数，TP通信域本卡ID。有TP域通信才需要传参。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：不支持TP域通信，使用默认值即可。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当有TP域通信时，取值范围\[0, 1\]，默认为0，同一个TP通信域中各卡的`tp_rank_id`不重复。无TP域通信时，传0即可。

-   **expert\_shard\_type** (`int`)：可选参数，表示共享专家卡排布类型。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：暂不支持该参数，使用默认值即可。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当前仅支持0，表示共享专家卡排在MoE专家卡前面。

-   **shared\_expert\_num** (`int`)：可选参数，表示共享专家数量，一个共享专家可以复制部署到多个卡上。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：暂不支持该参数，使用默认值即可。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围\[0, 4\]，0表示无共享专家，默认值为1。

-   **shared\_expert\_rank\_num** (`int`)：可选参数，表示共享专家卡数量。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：不支持共享专家，使用默认值即可。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围\[0, ep\_world\_size\)。取0表示无共享专家，不取0需满足shared\_expert\_rank\_num%shared\_expert\_num=0。

-   **quant\_mode** (`int`)：可选参数，表示量化模式。支持取值：0表示非量化（默认），2表示动态量化。当`quant_mode`为2，`dynamic_scales`不为None；当`quant_mode`为0，`dynamic_scales`为None。
-   **global\_bs** (`int`)：可选参数，表示EP域全局的batch size大小。当每个rank的BS不同时，支持传入max\_bs\*ep\_world\_size，其中max\_bs表示单rank BS最大值；当每个rank的BS相同时，支持取值0或BS\*ep\_world\_size。

-   **expert\_token\_nums\_type** (`int`)：可选参数，表示输出`expert_token_nums`的值类型，取值范围\[0, 1\]，0表示每个专家收到token数量的前缀和，1表示每个专家收到的token数量（默认）。

-   **comm\_alg** (`string`)：可选参数，表示通信亲和内存布局算法。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当前版本支持""，"fullmesh"，"hierarchy"三种输入方式。推荐配置"hierarchy"并搭配25.0.RC1.1及以上版本驱动使用。
        - "": 配置HCCL\_INTRA\_PCIE\_ENABLE=1和HCCL\_INTRA\_ROCE\_ENABLE=0时，调用"hierarchy"算法，否则调用"fullmesh"算法。不推荐使用该方式。
        - "fullmesh": token数据直接通过RDMA方式发往topk个目标专家所在的卡。
        - "hierarchy": token数据经过跨机、机内两次发送，仅不同server同号卡之间使用RDMA通信，server内使用HCCS通信。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：暂不支持该参数，使用默认值即可。

-   zero\_expert\_num：int类型，可选参数，表示零专家的数量。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：当前版本不支持，传0即可。</term>
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品：取值范围\[0, MAX_INT32\]，合法的零专家的ID值是\[moe\_expert\_num, moe\_expert\_num+zero\_expert\_num\)。</term>

-   copy\_expert\_num：int类型，可选参数，表示copy专家的数量。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：当前版本不支持，传0即可。</term>
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品：取值范围\[0, MAX_INT32\]，合法的零专家的ID值是\[moe\_expert\_num, moe\_expert\_num+zero\_expert\_num+copy\_expert\_num\)。</term>

-   const\_expert\_num：int类型，可选参数，表示常量专家的数量。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：当前版本不支持，传0即可。</term>
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品：取值范围\[0, MAX_INT32\]，合法的零专家的ID值是\[moe\_expert\_num, moe\_expert\_num+zero\_expert\_num+copy\_expert\_num+const\_expert\_num\)。</term>

## 输出说明<a name="zh-cn_topic_0000002203575833_section22231435517"></a>

-   **expand\_x** (`Tensor`)：表示本卡收到的token数据，要求为2维张量，shape为\(max\(tp\_world\_size, 1\) \*A, H\)，A表示在EP通信域可能收到的最大token数，数据类型支持`bfloat16`、`float16`、`int8`。量化时类型为`int8`，非量化时与`x`数据类型保持一致。数据格式为$ND$，支持非连续的Tensor。
-   **dynamic\_scales** (`Tensor`)：表示计算得到的动态量化参数。当`quant_mode`非0时才有该输出，要求为1维张量，shape为\(A,\)，数据类型支持`float`，数据格式支持$ND$，支持非连续的Tensor。
-   **assist\_info\_for\_combine** (`Tensor`)：表示给同一专家发送的token个数，要求是一个1维张量，shape为\(A \* 128, \)。数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_combine\_v2](torch_npu-npu_moe_distribute_combine_v2.md)的`assist_info_for_combine`输入。

-   **expert\_token\_nums** (`Tensor`)：本卡每个专家实际收到的token数量，要求为1维张量，shape为\(local\_expert\_num,\)，数据类型`int64`，数据格式支持$ND$，支持非连续的Tensor。
-   **ep\_recv\_counts** (`Tensor`)：表示EP通信域各卡收到的token数量，要求为1维张量，数据类型`int32`，数据格式支持$ND$，支持非连续的Tensor。对应[torch\_npu.npu\_moe\_distribute\_combine\_v2](torch_npu-npu_moe_distribute_combine_v2.md)的`ep_send_counts`输入。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：要求shape为\(moe\_expert\_num+2\*global\_bs\*K\*server\_num, \)，`global_bs`传入0时此处应当将其按照bs\*ep\_world\_size计算。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求shape为\(ep\_world\_size\*max\(tp\_world\_size, 1\)\*local\_expert\_num, \)。

-   **tp\_recv\_counts** (`Tensor`)：表示TP通信域各卡收到的token数量。对应[torch\_npu.npu\_moe\_distribute\_combine\_v2](torch_npu-npu_moe_distribute_combine_v2.md)的`tp_send_counts`输入。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：不支持TP通信域，暂无该输出，
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持TP通信域，要求是一个1D Tensor，shape为\(tp\_world\_size, \)，数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。

-   **expand\_scales** (`Tensor`)：表示`expert_scales`与`x`一起进行alltoallv之后的输出。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：要求是一个1维张量，shape为\(A, \)，数据类型支持`float`，数据格式要求为$ND$，支持非连续的Tensor。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：暂不支持该输出，返回None。

## 约束说明<a name="zh-cn_topic_0000002203575833_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持静态图模式（不低于PyTorch 2.1.0版本），并且`npu_moe_distribute_dispatch_v2`和`npu_moe_distribute_combine_v2`必须配套使用。
-   在不同产品型号、不同通信算法或不同版本中，`npu_moe_distribute_dispatch_v2`的Tensor输出`assist_info_for_combine`、`ep_recv_counts`、`tp_recv_counts`、`expand_scales`中的元素值可能不同，使用时直接将上述Tensor传给`npu_moe_distribute_combine_v2`对应参数即可，模型其他业务逻辑不应对其存在依赖。
-   调用接口过程中使用的`group_ep`、`ep_world_size`、`moe_expert_num`、`group_tp`、`tp_world_size`、`expert_shard_type`、`shared_expert_num`、`shared_expert_rank_num`、`global_bs`参数取值所有卡需保持一致，`group_ep`、`ep_world_size`、`group_tp`、`tp_world_size`、`expert_shard_type`、`global_bs`网络中不同层中也需保持一致，且和[torch\_npu.npu\_moe\_distribute\_combine\_v2](torch_npu-npu_moe_distribute_combine_v2.md)对应参数也保持一致。
-   动态缩容后的部署信息通过elastic_info参数传递给算子，无需修改其他参数。动态缩容后，MOE专家卡上的本卡部署MOE专家数需与缩容前保持一致。
-   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。
-   moe_expert_num + zero_expert_num + copy_expert_num + const_expert_num < MAX_INT32
-   参数里Shape使用的变量如下：
    -   A：表示本卡接收的最大token数量，取值范围如下
        -   不使能动态缩容场景时：
            -   对于共享专家，要满足A=BS\*shared\_expert\_num/shared\_expert\_rank\_num。
            -   对于MoE专家，当`global_bs`为0时，要满足A\>=BS\*ep\_world\_size\*min\(local\_expert\_num, K\)；当`global_bs`非0时，要满足A\>=global\_bs\* min\(local\_expert\_num, K\)。
        -   使能动态缩容场景时：
            -   当`global_bs`为0时，A>=max(BS\*ep_world_size\*shared_expert_num/shared_expert_rank_num, BS\*ep_world_size\*min(local_expert_num,K))；当`global_bs`非0时，A>=max(BS\*ep_world_size\*shared_expert_num/shared_expert_rank_num, global_bs*min(local_expert_num,K))

    -   H：表示hidden size隐藏层大小。
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：取值范围\(0,7168\]，且保证是32的整数倍。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值为\[1024, 8192\]。

    -   BS：表示batch sequence size，即本卡最终输出的token数量。
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：取值范围为0<BS≤256。
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围为0<BS≤512。

    -   K：表示选取topK个专家，取值范围为0<K≤16，同时满足0<K≤moe\_expert\_num。

    -   server\_num：表示服务器的节点数，取值只支持2、4、8。
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：仅该场景的shape使用了该变量。

    -   local\_expert\_num：表示本卡专家数量。
        -   对于共享专家卡，`local\_expert\_num`为1。
        -   对于MoE专家卡，local\_expert\_num=moe\_expert\_num/\(ep\_world\_size-shared\_expert\_rank\_num)，当`local_expert_num`大于1时，不支持TP域通信。

-   HCCL_BUFFSIZE:
    调用本接口前需检查HCCL\_BUFFSIZE环境变量取值是否合理该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：
        - comm\_alg配置为""或nullptr: 依照HCCL\_INTRA\_PCIE\_ENABLE和HCCL\_INTRA\_ROCE\_ENABLE配置选择"fullmesh"或"hierarchy"公式。
        - comm\_alg配置为"fullmesh": 要求\>=2\*\(BS\*ep\_world\_size\*min\(local\_expert\_num, K\)\*H\*sizeof\(unit16\)+2MB\)。
        - comm\_alg配置为"hierarchy": 要求=moe\_expert\_num\*BS\*\(H\*sizeof\(dtype_x\)+4\*\(\(K+7\)/8\*8\)\*sizeof\(uint32\)\)+4MB+100MB，不要求moe\_expert\_num\/\(ep\_world\_size - shared\_expert\_rank\_num\) <= 24。
        
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求 \>= 2且满足\>= 2 \* \(local\_expert\_num \* max\_bs \* ep\_world\_size \* Align512\(Align32\(2 \* H\) + 64\) + \(K + shared\_expert\_num\) \* max\_bs \* Align512\(2 \* H\)\)，local\_expert\_num需使用MoE专家卡的本卡专家数，其中Align512(x) = ((x+512-1)/512)\*512,Align32(x) = ((x+32-1)/32)\*32。

-   HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE:
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：该环境变量不再推荐使用，建议comm\_alg配置"hierarchy"。

-   本文公式中的“/”表示整除。

-   通信域使用约束：

    -   一个模型中的`npu_moe_distribute_dispatch_v2`和`npu_moe_distribute_combine_v2`算子仅支持相同EP通信域，且该通信域中不允许有其他算子。

    -   一个模型中的`npu_moe_distribute_dispatch_v2`和`npu_moe_distribute_combine_v2`算子仅支持相同TP通信域或都不支持TP通信域，有TP通信域时该通信域中不允许有其他算子。

-   版本配套约束：

     静态图模式下，从Ascend Extension for PyTorch 8.0.0版本开始，Ascend Extension for PyTorch框架会对静态图中最后一个节点输出结果做Meta推导与inferShape推导的结果强校验。当图中只有一个Dispatch\_v2算子，若CANN版本落后于Ascend Extension for PyTorch版本，会出现Shape不匹配报错，建议用户升级CANN版本，详细的版本配套关系参见《Ascend Extension for PyTorch 版本说明》中“<a href="https://www.hiascend.com/document/detail/zh/Pytorch/710/releasenote/releasenote_0003.html">相关产品版本配套说明</a>”。

## 调用示例<a name="zh-cn_topic_0000002203575833_section14459801435"></a>

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
    shared_expert_rank_num = 2                      # 共享专家数
    moe_expert_num = 14                            # moe专家数
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

    zero_expert_num = 0
    copy_expert_num = 0
    const_expert_num = 0

    elastic_ranks = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    def gen_elastic_info(elastic_ranks, is_elastic=1, actual_rank_num=14, elastic_shared_rank=4, elastic_moe_num=10, ep_world_size=16):
        if not is_elastic: return None
        elastic_info = torch.zeros(4 + 2 * ep_world_size, dtype=torch.int32)
        elastic_info[0] = is_elastic
        elastic_info[1] = actual_rank_num
        elastic_info[2] = shared_expert_rank_num
        elastic_info[3] = moe_expert_num
        table1 = [-1] * ep_world_size
        table2 = [-1] * ep_world_size

        for local_rank_id, ep_rank_id in enumerate(lastic_ranks):
            if ep_rank_id < ep_world_size:
                table1[ep_rank_id] = local_rank_id
                table2[local_rank_id] = ep_rank_id
        for i in range(ep_world_size):
            elastic_info[4 + i] = table1[i]
        for i in range(ep_world_size):
            elastic_info[4 + ep_world_size + i] = table2[i]
        assert elastic_info.shape[0] == 4 + 2 * ep_world_size
        if is_elastic:
            table1 = elastic_info[4:4+ep_world_size]
            table2 = elastic_info[4+ep_world_size:4+2*ep_world_size]
        return elastic_info

    def gen_const_expert_alpha_1():
        const_expert_alpha_1 = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1

    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_2

    def gen_const_expert_v():
        const_expert_v = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
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
    
    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        ep_group, tp_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        tp_hcomm_info = get_hcomm_info(rank, tp_group)
    
        # 创建输入tensor
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        expert_ids = gen_unique_topk_array(0, moeExpertNum+zero_expert_num+copy_expert_num+const_expert_num, bs, k).astype(np.int32)
        expert_ids = torch.from_numpy(expert_ids).npu()

        elastic_info= gen_elastic_info(elastic_ranks)
    
        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + moeExpertNum, h) if sharedExpertRankNum else (moeExpertNum, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None
    
        expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales = torch_npu.npu_moe_distribute_dispatch_v2(
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
            global_bs=globalBS,
            elastic_info=elastic_info,
            zero_expert_num=zero_expert_num,
            copy_expert_num=copy_expert_num,
            const_expert_num=const_expert_num)
        if is_quant:
            expand_x = expand_x.to(input_dtype)
        x = torch_npu.npu_moe_distribute_combine_v2(expand_x=expand_x,
                                                 expert_ids=expert_ids,
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
                                                 global_bs=globalBS,
                                                 elastic_info=elastic_info,
                                                 ori_x=x,
                                                 const_expert_alpha_1=const_expert_alpha_1,
                                                 const_expert_alpha_2=const_expert_alpha_2,
                                                 const_expert_v=const_expert_v,
                                                 zero_expert_num=zero_expert_num,
                                                 copy_expert_num=copy_expert_num,
                                                 const_expert_num=const_expert_num)
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

    zero_expert_num = 0
    copy_expert_num = 0
    const_expert_num = 0

    elastic_ranks = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    def gen_elastic_info(elastic_ranks, is_elastic=1, actual_rank_num=14, elastic_shared_rank=4, elastic_moe_num=10, ep_world_size=16):
        if not is_elastic: return None
        elastic_info = torch.zeros(4 + 2 * ep_world_size, dtype=torch.int32)
        elastic_info[0] = is_elastic
        elastic_info[1] = actual_rank_num
        elastic_info[2] = shared_expert_rank_num
        elastic_info[3] = moe_expert_num
        table1 = [-1] * ep_world_size
        table2 = [-1] * ep_world_size

        for local_rank_id, ep_rank_id in enumerate(lastic_ranks):
            if ep_rank_id < ep_world_size:
                table1[ep_rank_id] = local_rank_id
                table2[local_rank_id] = ep_rank_id
        for i in range(ep_world_size):
            elastic_info[4 + i] = table1[i]
        for i in range(ep_world_size):
            elastic_info[4 + ep_world_size + i] = table2[i]
        assert elastic_info.shape[0] == 4 + 2 * ep_world_size
        if is_elastic:
            table1 = elastic_info[4:4+ep_world_size]
            table2 = elastic_info[4+ep_world_size:4+2*ep_world_size]
        return elastic_info

    def gen_const_expert_alpha_1():
        const_expert_alpha_1 = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1

    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_2

    def gen_const_expert_v():
        const_expert_v = torch.empty(size=[const_expert_num], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_v
    
    class MOE_DISTRIBUTE_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, x, expert_ids, group_ep, group_tp, ep_world_size, tp_world_size,
                    ep_rank_id, tp_rank_id, expert_shard_type, shared_expert_rank_num, moe_expert_num,
                    scales, quant_mode, global_bs, expert_scales, elastic_info, x, const_expert_alpha_1, const_expert_alpha_2, const_expert_v, zero_expert_num, copy_expert_num, const_expert_num):
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
                                                                      global_bs=global_bs,
                                                                      elastic_info=elastic_info,
                                                                      ori_x=x,
                                                                      const_expert_alpha_1=const_expert_alpha_1,
                                                                      const_expert_alpha_2=const_expert_alpha_2,
                                                                      const_expert_v=const_expert_v,
                                                                      zero_expert_num=zero_expert_num,
                                                                      copy_expert_num=copy_expert_num,
                                                                      const_expert_num=const_expert_num)
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
        expert_ids = gen_unique_topk_array(0, moeExpertNum+zero_expert_num+copy_expert_num+const_expert_num, bs, k).astype(np.int32)
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
                               quant_mode, globalBS, expert_scales, elastic_info, x, const_expert_alpha_1, const_expert_alpha_2, const_expert_v, zero_expert_num, copy_expert_num, const_expert_num)
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

