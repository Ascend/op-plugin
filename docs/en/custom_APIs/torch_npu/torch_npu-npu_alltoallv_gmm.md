# torch\_npu.npu\_alltoallv\_gmm<a name="en-us_topic_0000002350725076"></a>

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |

## Function<a name="en-us_topic_0000002282815538_section14441124184110"></a>

- Description: Fuses `AlltoAllv`, `Permute`, and `GroupedMatMul` for routed experts in an MoE network, implements parallel fused computation with the shared expert `MatMul`, and uses a communication-before-computation sequence.

- Routed expert formula:

    ![](../../figures/en-us_formulaimage_0000002357766385.png)

    - `ata_out` is the output tensor of the `AlltoAllv` communication applied to `gmm_x`, used as the input for `Permute`.
    - `permute_out` is the output tensor of the `Permute` operation, serving as the left matrix in the routed expert GroupedMatMul computation.
    - `gmm_weight` represents the right matrix for the GroupedMatMul computation by the routed expert.
    - `gmm_y` represents the final output tensor of the routed expert `GroupedMatMul` computation.

- Shared expert formula:

    ![](../../figures/en-us_formulaimage_0000002323547858.png)

    - `mm_x` indicates the left matrix for shared expert MatMul.
    - `mm_weight` indicates the right matrix for shared expert MatMul.
    - `mm_y` indicates the output of shared expert MatMul.

## Prototype<a name="en-us_topic_0000002282815538_section45077510411"></a>

```python
torch_npu.npu_alltoallv_gmm(gmm_x, gmm_weight, hcom, ep_world_size, send_counts, recv_counts, *, send_counts_tensor=None, recv_counts_tensor=None, mm_x=None, mm_weight=None, trans_gmm_weight=False, trans_mm_weight=False, permute_out_flag=False) -> (Tensor, Tensor, Tensor)
```

## Parameters<a name="en-us_topic_0000002282815538_section112637109429"></a>

- **`gmm_x`** (`Tensor`): Required. Result of AlltoAllv communication and Permute operations, used as the left matrix for the GroupedMatMul computation. The data type can be `float16` or `bfloat16`. This parameter must be 2D with shape `(BSK, H1)`. The data layout can be ND.
- **`gmm_weight`** (`Tensor`): Required. Right matrix for GroupedMatMul. The data type must be identical to that of `gmm_x`. This parameter must be 3D with shape `(e, H1, N1)`. The data layout can be ND.
- **`hcom`** (`str`): Required. Communicator name for expert parallelism. The string length must fall within the range of (0, 128).
- **`ep_world_size`** (`int`): Required. Size of the EP communication domain. Valid values: `8`, `16`, `32`, `64`, or `128`.
- **`send_counts`** (`List[int]`): Required. Number of tokens sent to other devices. The list length is `e * ep_world_size`, and the maximum value is `256`.
- **`recv_counts`** (`List[int]`): Required. Number of tokens received from other devices. The list length is `e * ep_world_size`, and the maximum value is 256.
- **`send_counts_tensor`** (`Tensor`): Optional. The data type can be `int`. The shape of this parameter is `(e * ep_world_size,)`. The data layout can be ND. **This parameter is not supported in the current version**. Retain the default value.
- **`recv_counts_tensor`** (`Tensor`): Optional. The data type can be `int`. The shape of this parameter is `(e * ep_world_size,)`. The data layout can be ND. **This parameter is not supported in the current version**. Retain the default value.
- **`mm_x`** (`Tensor`): Optional. Left matrix for shared expert MatMul. This parameter must be provided when shared expert computation is fused. The data type can be `float16` or `bfloat16`. This parameter must be a 2D tensor, with shape `(BS, H2)`.
- **`mm_weight`** (`Tensor`): Optional. Right matrix for shared expert MatMul. This parameter must be provided when shared expert computation is fused. The data type must be identical to that of `mm_x`. This parameter must be a 2D tensor, with shape `(H2, N2)`.
- **`trans_gmm_weight`** (`bool`): Optional. Specifies whether to transpose the right matrix of GroupedMatMul. `True` enables transposition and `False` disables it.
- **`trans_mm_weight`** (`bool`): Optional. Specifies whether to transpose the right matrix of shared expert MatMul. `True` enables transposition and `False` disables it.
- **`permute_out_flag`** (`bool`): Optional. Specifies whether to output the result of the Permute operation. Valid values are `True` (outputs the Permute result) or `False` (does not output it).

## Return Values<a name="en-us_topic_0000002282815538_section22231435517"></a>

- **`gmm_y`** (`Tensor`): Computation output, representing the final result of GroupedMatMul. The data type is identical to that of `gmm_x`. This parameter must be 2D with shape `(A, N1)`.
- **`mm_y`** (`Tensor`): Computation output of the shared expert MatMul. The data type is identical to `mm_x`. This parameter must be 2D with shape `(BS, N2)`. This output is generated only when `mm_x` and `mm_weight` are provided.
- **`permute_out`** (`Tensor`): Output of the Permute operation. The data type is identical to that of `gmm_x`.

## Constraints<a name="en-us_topic_0000002282815538_section12345537164214"></a>

- This API can be used in inference scenarios.
- This API supports graph mode.
- The communication volume per rank must be greater than or equal to `2` MB.
- Variables used in input parameter tensor shapes:
    - `BSK`: Number of tokens sent by the current rank (`BS * K = BSK`), equal to the sum of `send_counts`. The range is (0, 52428800).
    - `H1`: Hidden layer size of the routed experts. The value range is (0, 65536).

    - `H2`: Hidden layer size of the shared experts. The value range is (0, 12288].
    - `e`: Number of experts on a single rank, which must be less than or equal to 32. The maximum supported value for `e * ep_world_size` is 256.

    - `N1`: Number of heads for routed experts. The value range is (0, 65536).
    - `N2`: Number of heads for shared experts. The value range is (0, 65536).

    - `BS`: Batch sequence size.
    - `K`: Number of top-K experts selected. The value range is [2, 8].

    - `A`: Number of tokens received by the current rank, which is the sum of `recv_counts`.
    - The sum of `A` across all ranks in the EP communication domain must be identical to the sum of `BSK` across all ranks.

## Examples<a name="en-us_topic_0000002282815538_section14459801435"></a>

- Single-operator call

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

- Graph mode call

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
        # Static graph: dynamic=False; Dynamic graph: dynamic=True.
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
