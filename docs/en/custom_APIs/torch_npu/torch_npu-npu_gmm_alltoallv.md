# torch_npu.npu_gmm_alltoallv

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->

## Function

- **Function**: In an MoE network, this API fuses the GroupedMatMul and AlltoAllv operations of routed experts and implements parallel fusion with the shared-expert MatMul. Computation is performed before communication.

- **Formulas**:
    - Routed experts:
        $$
        gmm\_y = gmm\_x \times gmm\_weight
        $$

        $$
        unpermute\_out = Unpermute(gmm\_y)
        $$

        $$
        y = AlltoAllv(unpermute\_out)
        $$

        - gmm\_x indicates the left matrix of the routed-expert GroupedMatMul computation.
        - gmm\_weight indicates the right matrix of the routed-expert GroupedMatMul computation.
        - gmm\_y indicates the output of the routed-expert GroupedMatMul computation, which is used for the subsequent Unpermute computation.
        - unpermute\_out is the output of the Unpermute computation on gmm\_y, which serves as the input of the AlltoAllv communication.
        - y indicates the output of the AlltoAllv communication on unpermute\_out.

    - Shared experts:
        $$
        mm\_y = mm\_x \times mm\_weight
        $$

        - mm\_x indicates the left matrix of the shared-expert MatMul computation.
        - mm\_weight indicates the right matrix of the shared-expert MatMul computation.
        - mm\_y indicates the output of the shared-expert MatMul computation.

## Prototype

```python
torch_npu.npu_gmm_alltoallv(gmm_x, gmm_weight, hcom, ep_world_size, send_counts, recv_counts, *, send_counts_tensor=None, recv_counts_tensor=None, mm_x=None, mm_weight=None, trans_gmm_weight=False, trans_mm_weight=False, comm_mode=None) -> (Tensor, Tensor)
```

## Parameters

- **`gmm_x`** (`Tensor`): **Required**. Left matrix of the GroupedMatMul computation. The data type can be `torch.float16` or `torch.bfloat16`. Two dimensions are supported, the shape is (A, H1), and the data layout can be $ND$.
- **`gmm_weight`** (`Tensor`): **Required**. Right matrix of the GroupedMatMul computation. The data type must be the same as that of `gmm_x`. Three dimensions are supported, the shape is (e, H1, N1), and the data layout can be $ND$.
- **`hcom`** (`str`): Required. Communicator name for expert parallelism. The string length must fall within the range of (0, 128).
- **`ep_world_size`** (`int`): **Required**. Size of the EP communication domain. CCU supports only intra-node UB domain interconnection, while the AI CPU supports inter-node UB domain interconnection.

    <!-- npu="A3" id3 -->
    - <term>Atlas A3 products</term>: the supported values are 8, 16, 32, 64, and 128.
    <!-- end id3 -->
    <!-- npu="950" id4 -->
    - <term>Ascend 950DT products</term>: the supported values are 2, 4, 8, 16, 32, and 64.
    <!-- end id4 -->

- **`send_counts`** (`List[int]`): **Required**. Number of tokens sent to other devices. The data type can be int. The value size is e\*`ep_world_size` and the maximum is 256. The input type must be list.
- **`recv_counts`** (`List[int]`): **Required**. Number of tokens received from other devices. The data type can be int. The value size is e\*`ep_world_size` and the maximum is 256. The input type must be list.
- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`send_counts_tensor`** (`Tensor`): **Optional**. The data type can be `torch.int`, the shape is (e\*ep\_world\_size,), and the data layout can be $ND$. **This parameter is not supported in the current version.** Use the default value.
- **`recv_counts_tensor`** (`Tensor`): **Optional**. The data type can be `torch.int`, the shape is (e\*ep\_world\_size,), and the data layout can be $ND$. **This parameter is not supported in the current version.** Use the default value.
- **`mm_x`** (`Tensor`): **Optional**. Left matrix in the shared-expert MatMul computation. When the shared-expert matrix computation must be fused, this parameter is required. Two dimensions are supported, and the shape is (BS, H2).
- **`mm_weight`** (`Tensor`): **Optional**. Right matrix in the shared-expert MatMul computation. When the shared-expert matrix computation must be fused, this parameter is required. Two dimensions are supported, and the shape is (H2, N2).
- **`trans_gmm_weight`** (`bool`): **Optional**. Specifies whether the right matrix of the GroupedMatMul needs to be transposed. `true` indicates that it needs to be transposed, and `false` indicates that it does not.
- **`trans_mm_weight`** (`bool`): **Optional**. Specifies whether the right matrix of the shared-expert MatMul needs to be transposed. `true` indicates that it needs to be transposed, and `false` indicates that it does not.
- **`comm_mode`** (`str`): **Optional**. Communication engine mode. The default value is `None`.

    <!-- npu="A3" id5 -->
    - <term>Atlas A3 products</term>: the supported values are `None` and `ai_cpu`. If any value is passed, AI CPU communication is used.
    <!-- end id5 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950DT products</term>: the supported values are `None`, `ai_cpu`, and `ccu`. When the value is `None`, AI CPU communication is used.
    <!-- end id6 -->

## Return Values

- **`y`** (`Tensor`): Final computation result. The data type must be identical to that of `gmm_x`. This parameter must be 2D with shape `(BSK, N1)`.
- **`mm_y`** (`Tensor`): Output of the shared-expert MatMul. The data type is the same as that of `mm_x`. Two dimensions are supported, and the shape is (BS, N2). It is output only when `mm_x` and `mm_weight` are passed.

## Constraints

- This API can be used in inference scenarios.
- This API supports single-operator mode and TorchAir graph mode.
- **Single-device communication volume requirement**:

  <!-- npu="A3" id7 -->
  - <term>Atlas A3 products</term>: the single-device communication volume must be greater than or equal to 2 MB.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950DT products</term>: there is no constraint on the single-device communication volume.
  <!-- end id8 -->

- Description of the variables used in the shapes of the input parameter Tensors:
  - BSK: number of tokens received by the current device (BS\*K=BSK), which is the accumulated sum of the recv\_counts parameters. The value range is (0, 52428800).
  - H1: hidden layer size of routed experts. The value range is (0, 65536).
  - H2: hidden layer size of shared experts. The value range is (0, 12288].
  - e: number of experts on a single device. e<=32, and the maximum supported value of e\*ep\_world\_size is 256.
  - N1: head\_num of routed experts. The value range is (0, 65536).
  - N2: head\_num of shared experts. The value range is (0, 65536).
  - BS: batch sequence size.
  - K: number of top\_k experts selected. The value range of K is [2, 8].
  - A: number of tokens sent by the current device, which is the accumulated sum of the send\_counts parameters.
  - The accumulated sum of the A parameters on all devices in the EP communication domain is equal to the accumulated sum of the BSK parameters on all devices.

## Example

- Single-operator call

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

- Graph mode call

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
        # Static graph: dynamic=False; dynamic graph: dynamic=True
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
