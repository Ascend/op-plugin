# torch\_npu.npu\_alltoallv\_gmm<a name="ZH-CN_TOPIC_0000002350725076"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |

## 功能说明<a name="zh-cn_topic_0000002282815538_section14441124184110"></a>

- API功能：MoE（Mixture of Experts，混合专家模型）网络中，完成路由专家AlltoAllv、Permute、GroupedMatMul融合并实现与共享专家MatMul并行融合，先通信后计算。

-   路由专家计算公式：

    ![](figures/zh-cn_formulaimage_0000002357766385.png)

    -   ata\_out是gmm\_x进行AlltoAllv通信的输出结果，后续用于Permute计算。
    -   permute\_out是ata\_out进行Permute计算的输出结果，作为路由专家进行GroupedMatMul计算的左矩阵。
    -   gmm\_weight指路由专家进行GroupedMatMul计算的右矩阵。
    -   gmm\_y指路由专家进行GroupedMatMul计算的输出。

-   共享专家计算公式：

    ![](figures/zh-cn_formulaimage_0000002323547858.png)

    -   mm\_x指共享专家MatMul计算的左矩阵。
    -   mm\_weight指共享专家MatMul计算的右矩阵。
    -   mm\_y指共享专家MatMul计算的输出。

## 函数原型<a name="zh-cn_topic_0000002282815538_section45077510411"></a>

```
torch_npu.npu_alltoallv_gmm(gmm_x, gmm_weight, hcom, ep_world_size, send_counts, recv_counts, *, send_counts_tensor=None, recv_counts_tensor=None, mm_x=None, mm_weight=None, trans_gmm_weight=False, trans_mm_weight=False, permute_out_flag=False) -> (Tensor, Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002282815538_section112637109429"></a>

-   **gmm\_x**（`Tensor`）：必选参数，AlltoAllv通信与Permute操作后结果作为GroupedMatMul计算的左矩阵。数据类型支持`float16`、`bfloat16`，支持2维，shape为$(BSK, H1)$，数据格式支持ND。
-   **gmm\_weight**（`Tensor`）：必选参数，GroupedMatMul计算的右矩阵。数据类型与`gmm_x`保持一致，支持3维，shape为$(e, H1, N1)$，数据格式支持ND。
-   **hcom**（`str`）：必选参数，专家并行的通信域名，字符串长度要求\(0, 128\)。
-   **ep\_world\_size**（`int`）：必选参数，EP通信域size，取值支持8、16、32、64、128。
-   **send\_counts**（`List[int]`）：必选参数，表示发送给其他卡的token数，数据类型支持int，取值大小为e\*`ep_world_size`，最大为256。
-   **recv\_counts**（`List[int]`）：必选参数，表示接收其他卡的token数，数据类型支持int，取值大小为e\*`ep_world_size`，最大为256。
-   **send\_counts\_tensor**（`Tensor`）：可选参数，数据类型支持int，shape为$(e*ep\_world\_size,)$，数据格式支持ND。**当前版本暂不支持**，使用默认值即可。
-   **recv\_counts\_tensor**（`Tensor`）：可选参数，数据类型支持int，shape为$(e*ep\_world\_size,)$，数据格式支持ND。**当前版本暂不支持**，使用默认值即可。
-   **mm\_x**（`Tensor`）：可选参数，共享专家MatMul计算中的左矩阵。当需要融合共享专家矩阵计算时，该参数必选，数据类型支持`float16`、`bfloat16`，支持2维，shape为$(BS, H2)$。
-   **mm\_weight**（`Tensor`）：可选参数，共享专家MatMul计算中的右矩阵。当需要融合共享专家矩阵计算时，该参数必选，数据类型与`mm_x`保持一致，支持2维，shape为$(H2, N2)$。
-   **trans\_gmm\_weight**（`bool`）：可选参数，GroupedMatMul的右矩阵是否需要转置，true表示需要转置，false表示不转置。
-   **trans\_mm\_weight**（`bool`）：可选参数，共享专家MatMul的右矩阵是否需要转置，true表示需要转置，false表示不转置。
-   **permute\_out\_flag**（`bool`）：可选参数，Permute结果是否需要输出，true表明需要输出，false表明不需要输出。

## 返回值说明<a name="zh-cn_topic_0000002282815538_section22231435517"></a>

-   **gmm\_y**（`Tensor`）：计算输出，表示最终的计算结果，数据类型与输入`gmm_x`保持一致，支持2维，shape为$(A, N1)$。
-   **mm\_y**（`Tensor`）：计算输出，共享专家MatMul的输出，数据类型与`mm_x`保持一致，支持2维，shape为$(BS, N2)$。仅当传入`mm_x`与`mm_weight`才输出。
-   **permute\_out**（`Tensor`）：计算输出，Permute之后的输出，数据类型与`gmm_x`保持一致。

## 约束说明<a name="zh-cn_topic_0000002282815538_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   单卡通信量取值大于等于2MB。
-   输入参数Tensor中shape使用的变量说明：
    -   BSK：本卡发送的token数（BS\*K=BSK），是send\_counts参数累加之和，取值范围\(0, 52428800\)。
    -   H1：表示路由专家hidden size隐藏层大小，取值范围\(0, 65536\)。

    -   H2：表示共享专家hidden size隐藏层大小，取值范围\(0, 12288\]。
    -   e：表示单卡上专家个数，e<=32，e\*ep\_world\_size最大支持256。

    -   N1：表示路由专家的head\_num，取值范围\(0, 65536\)。
    -   N2：表示共享专家的head\_num，取值范围\(0, 65536\)。

    -   BS：表示batch sequence size。
    -   K：表示选取topK个专家，K的范围\[2, 8\]。

    -   A：本卡收到的token数，是recv\_counts参数累加之和。
    -   EP通信域内所有卡上的A累加和等于所有卡上的BSK累加和。

## 调用示例<a name="zh-cn_topic_0000002282815538_section14459801435"></a>

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    
    def run_npu_alltoallv_gmm(rank, ep_world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype):
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
    
        print(torch_npu.npu_alltoallv_gmm(gmm_x =input,
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
                                                trans_mm_weight  = False,
                                                permute_out_flag  = False))
    
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
        mp.spawn(run_npu_alltoallv_gmm, args=(epWorkSize, master_ip, master_port, gmm_x_shape, gmm_weight_shape, send_counts, recv_counts, dtype), nprocs=epWorkSize)
    ```

-   图模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torchair
    
    class ALLTOALLV_GMM_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self,gmm_x, gmm_weight,
                    hcom, ep_world_size,
                    send_counts, recv_counts, send_counts_tensor, recv_counts_tensor,
                    mm_x, mm_weight,
                    trans_gmm_weight, trans_mm_weight, permute_out_flag):
            return torch_npu.npu_alltoallv_gmm(gmm_x =gmm_x,
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
                                                trans_mm_weight  = trans_mm_weight,
                                                permute_out_flag  = permute_out_flag)
    
    def run_npu_alltoallv_gmm(rank, ep_world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype):
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
    
        model = ALLTOALLV_GMM_GRAPH_Model()
        npu_backend = torchair.get_npu_backend(compiler_config=None)
        # 静态图：dynamic=False；动态图：dynamic=True
        model = torch.compile(ALLTOALLV_GMM_GRAPH_Model(), backend=npu_backend, dynamic=False)
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
                        trans_mm_weight=False,
                        permute_out_flag=True))
    
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
        mp.spawn(run_npu_alltoallv_gmm, args=(epWorkSize, master_ip, master_port, gmm_x_shape, gmm_weight_shape, send_counts, recv_counts, dtype), nprocs=epWorkSize)
    ```

