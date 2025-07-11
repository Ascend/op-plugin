# torch\_npu.npu\_gmm\_alltoallv<a name="ZH-CN_TOPIC_0000002384445761"></a>

## 产品支持情况<a name="zh-cn_topic_0000002317314449_section1369303644412"></a>

<a name="zh-cn_topic_0000002317314449_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002317314449_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002317314449_p1883113061818"><a name="zh-cn_topic_0000002317314449_p1883113061818"></a><a name="zh-cn_topic_0000002317314449_p1883113061818"></a><span id="zh-cn_topic_0000002317314449_ph24751558184613"><a name="zh-cn_topic_0000002317314449_ph24751558184613"></a><a name="zh-cn_topic_0000002317314449_ph24751558184613"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002317314449_p783113012187"><a name="zh-cn_topic_0000002317314449_p783113012187"></a><a name="zh-cn_topic_0000002317314449_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002317314449_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002317314449_p2098311377352"><a name="zh-cn_topic_0000002317314449_p2098311377352"></a><a name="zh-cn_topic_0000002317314449_p2098311377352"></a><span id="zh-cn_topic_0000002317314449_ph1719614396352"><a name="zh-cn_topic_0000002317314449_ph1719614396352"></a><a name="zh-cn_topic_0000002317314449_ph1719614396352"></a><term id="zh-cn_topic_0000002317314449_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002317314449_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002317314449_zh-cn_topic_0000001312391781_term1253731311225"></a><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002317314449_p7948163910184"><a name="zh-cn_topic_0000002317314449_p7948163910184"></a><a name="zh-cn_topic_0000002317314449_p7948163910184"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0000002317314449_section14441124184110"></a>

-   算子功能：MoE网络中，完成路由专家GroupedMatMul、AlltoAllv融合并实现与共享专家MatMul并行融合，先计算后通信。
-   路由专家计算公式：

    ![](figures/zh-cn_formulaimage_0000002323688460.png)

    -   gmm\_x指路由专家GroupedMatMul计算的左矩阵。
    -   gmm\_weight指路由专家GroupedMatMul计算的右矩阵。
    -   gmm\_y指路由专家进行GroupedMatMul计算的输出，后续用于Unpermute计算。
    -   unpermute\_out是gmm\_y进行Unpermute计算的输出结果，作为AlltoAllv通信的输入。
    -   y指对unpermute\_out进行AlltoAllv通信输出。

-   共享专家计算公式：

    ![](figures/zh-cn_formulaimage_0000002323838248.png)

    -   mm\_x指共享专家MatMul计算的左矩阵。
    -   mm\_weight指共享专家MatMul计算的右矩阵。
    -   mm\_y指共享专家MatMul计算的输出。

## 函数原型<a name="zh-cn_topic_0000002317314449_section45077510411"></a>

```
torch_npu.npu_gmm_alltoallv(Tensor gmm_x, Tensor gmm_weight, str hcom, int ep_world_size, int[] send_counts, int[] recv_counts, *, Tensor? send_counts_tensor=None, Tensor? recv_counts_tensor=None, Tensor? mm_x=None, Tensor? mm_weight=None, bool trans_gmm_weight=False, bool trans_mm_weight=False) -> (Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002317314449_section112637109429"></a>

-   **gmm\_x**（Tensor）：必选参数，GroupedMatMul计算的左矩阵。数据类型支持float16、bfloat16，支持2维，shape为\(A, H1\)，数据格式支持ND。
-   **gmm\_weight**（Tensor）：必选参数，GroupedMatMul计算的右矩阵。数据类型与gmm\_x保持一致，支持3维，shape为\(e, H1, N1\)，数据格式支持ND。
-   **hcom**（str）：必选参数，专家并行的通信域名，字符串长度要求\(0, 128\)。
-   **ep\_world\_size**（int）：必选参数，EP通信域size，取值支持8、16、32、64。
-   **send\_counts**（int\[\]）：必选参数，表示发送给其他卡的token数，数据类型支持int，取值大小为e\*ep\_world\_size，最大为256。
-   **recv\_counts**（int\[\]）：必选参数，表示接收其他卡的token数，数据类型支持int，取值大小为e\*ep\_world\_size，最大为256。
-   **send\_counts\_tensor**（Tensor）：可选参数，数据类型支持int，shape为\(e\*ep\_world\_size,\)，数据格式支持ND。**当前版本暂不支持**，使用默认值即可。
-   **recv\_counts\_tensor**（Tensor）：可选参数，数据类型支持int，shape为\(e\*ep\_world\_size,\)，数据格式支持ND。**当前版本暂不支持**，使用默认值即可。
-   **mm\_x**（Tensor）：可选参数，共享专家MatMul计算中的左矩阵。当需要融合共享专家矩阵计算时，该参数必选，支持2维，shape为\(BS, H2\)。
-   **mm\_weight**（Tensor）：可选参数，共享专家MatMul计算中的右矩阵。当需要融合共享专家矩阵计算时，该参数必选，支持2维，shape为\(H2, N2\)。
-   **trans\_gmm\_weight**（bool）：可选参数，GroupedMatMul的右矩阵是否需要转置，true表示需要转置，false表示不转置。
-   **trans\_mm\_weight**（bool）：可选参数，共享专家MatMul的右矩阵是否需要转置，true表示需要转置，false表示不转置。

## 返回值说明<a name="zh-cn_topic_0000002317314449_section22231435517"></a>

-   **y**（Tensor）：表示最终计算结果，数据类型与输入gmm\_x保持一致，支持2维，shape为\(BSK, N1\)。
-   **mm\_y**（Tensor）：共享专家MatMul的输出，数据类型与mm\_x保持一致，支持2维，shape为\(BS, N2\)。仅当传入mm\_x与mm\_weight才输出。

## 约束说明<a name="zh-cn_topic_0000002317314449_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。
-   单卡通信量要求在2MB到100MB范围内。
-   输入参数Tensor中shape使用的变量说明：
    -   BSK：本卡接收的token数（BS\*K=BSK），是recv\_counts参数累加之和，取值范围\(0, 52428800\)。

    -   H1：表示路由专家hidden size隐藏层大小，取值范围\(0, 65536\)。
    -   H2：表示共享专家hidden size隐藏层大小，取值范围\(0, 12288\]。
    -   e：表示单卡上专家个数，e<=32，e \* ep\_world\_size最大支持256。
    -   N1：表示路由专家的head\_num，取值范围\(0, 65536\)。
    -   N2：表示共享专家的head\_num，取值范围\(0, 65536\)。
    -   BS：batch sequence size。
    -   K：表示选取top\_k个专家，K的范围\[2, 8\]。
    -   A：本卡发送的token数，是send\_counts参数累加之和。
    -   EP通信域内所有卡上的A参数的累加和等于所有卡上的BSK参数的累加和。

## 调用示例<a name="zh-cn_topic_0000002317314449_section14459801435"></a>

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    
    def run_npu_gmm_alltoallv(rank, ep_world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=ep_world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
    
        input = torch.randn(gmm_x, dtype=dtype).npu()
        weight = torch.randn(gmm_w, dtype=dtype).npu()
    
        print(torch_npu.npu_gmm_alltoallv(gmm_x =input,
                                                gmm_weight = weight,
                                                hcom= hcom_info,
                                                ep_world_size = ep_world_size,
                                                send_counts = list(send_counts),
                                                recv_counts = list(recv_counts),
                                                send_counts_tensor = None, #send_counts_tensor,
                                                recv_counts_tensor = None, #recv_counts_tensor,
                                                mm_x = None,
                                                mm_weight = None,
                                                trans_gmm_weight = False,
                                                trans_mm_weight  = False))
    
    if __name__ == "__main__":
        epWorkSize = 8
        e = 4
        master_ip = '127.0.0.1'
        master_port = '50001'
        BS = 512
        K = 8
        gmm_x_shape = [BS*K, 2048]
        gmm_weight_shape = [e, 2048, 2048]
        send_counts = [128] * (e * epWorkSize)
        recv_counts = [128] * (e * epWorkSize)
    
        dtype = torch.float16
    
        mp.spawn(run_npu_gmm_alltoallv, args=(epWorkSize, master_ip, master_port, gmm_x_shape, gmm_weight_shape, send_counts, recv_counts, dtype), nprocs=epWorkSize)
    ```

-   图模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torchair
    
    class GMM_ALLTOALLV_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self,gmm_x, gmm_weight,
                    hcom, ep_world_size,
                    send_counts, recv_counts, send_counts_tensor, recv_counts_tensor,
                    mm_x, mm_weight,
                    trans_gmm_weight, trans_mm_weight):
            return torch_npu.npu_gmm_alltoallv(gmm_x =gmm_x,
                                                gmm_weight = gmm_weight,
                                                hcom= hcom,
                                                ep_world_size = ep_world_size,
                                                send_counts = list(send_counts),
                                                recv_counts = list(recv_counts),
                                                send_counts_tensor = None, #send_counts_tensor,
                                                recv_counts_tensor = None, #recv_counts_tensor,
                                                mm_x = mm_x,
                                                mm_weight = mm_weight,
                                                trans_gmm_weight = trans_gmm_weight,
                                                trans_mm_weight  = trans_mm_weight)
    
    def run_npu_gmm_alltoallv(rank, ep_world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=ep_world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
    
        input = torch.randn(gmm_x, dtype=dtype).npu()
        weight = torch.randn(gmm_w, dtype=dtype).npu()
    
        model = GMM_ALLTOALLV_GRAPH_Model()
        npu_backend = torchair.get_npu_backend(compiler_config=None)
        # 静态图：dynamic=False；动态图：dynamic=True
        model = torch.compile(GMM_ALLTOALLV_GRAPH_Model(), backend=npu_backend, dynamic=False)
        print(model(gmm_x=input,
                        gmm_weight=weight,
                        send_counts_tensor=None,
                        recv_counts_tensor=None,
                        mm_x=None,
                        mm_weight=None,
                        hcom=hcom_info,
                        ep_world_size=ep_world_size,
                        send_counts=send_counts,
                        recv_counts=recv_counts,
                        trans_gmm_weight=False,
                        trans_mm_weight=False))
    
    if __name__ == "__main__":
        epWorkSize = 8
        e = 4
        master_ip = '127.0.0.1'
        master_port = '50001'
        BS = 512
        K = 8
        gmm_x_shape = [BS*K, 2048]
        gmm_weight_shape = [e, 2048, 2048]
        send_counts = [128] * (e * epWorkSize)
        recv_counts = [128] * (e * epWorkSize)
    
        dtype = torch.float16
    
        mp.spawn(run_npu_gmm_alltoallv, args=(epWorkSize, master_ip, master_port, gmm_x_shape, gmm_weight_shape, send_counts, recv_counts, dtype), nprocs=epWorkSize)
    ```

