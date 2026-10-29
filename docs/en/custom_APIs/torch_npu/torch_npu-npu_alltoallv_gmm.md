# torch_npu.npu_alltoallv_gmm

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->

## Function

- **Function**: In an MoE (mixture of experts) network, this API fuses the routed-expert AlltoAllv, Permute, and GroupedMatMul operations and implements parallel fusion with the shared-expert MatMul. Communication is performed before computation.

- **Formulas**:
    - Routed experts:
        $$
        ata\_out = AlltoAllv(gmm\_x)
        $$

        $$
        permute\_out = Permute(ata\_out)
        $$

        $$
        gmm\_y = permute\_out \times gmm\_weight
        $$

        - ata\_out is the output of the AlltoAllv communication on gmm\_x, which is used for the subsequent Permute computation.
        - permute\_out is the output of the Permute computation on ata\_out, which serves as the left matrix of the GroupedMatMul computation of routed experts.
        - gmm\_weight indicates the right matrix of the GroupedMatMul computation of routed experts.
        - gmm\_y indicates the output of the GroupedMatMul computation of routed experts.

    - Shared experts:
        $$
        mm\_y = mm\_x \times mm\_weight
        $$

        - mm\_x indicates the left matrix of the shared-expert MatMul computation.
        - mm\_weight indicates the right matrix of the shared-expert MatMul computation.
        - mm\_y indicates the output of the shared-expert MatMul computation.

## Prototype

```python
torch_npu.npu_alltoallv_gmm(gmm_x, gmm_weight, hcom, ep_world_size, send_counts, recv_counts, *, send_counts_tensor=None, recv_counts_tensor=None, mm_x=None, mm_weight=None, trans_gmm_weight=False, trans_mm_weight=False, permute_out_flag=False, comm_mode=None) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`gmm_x`** (`Tensor`): Required. Result of AlltoAllv communication and Permute operations, used as the left matrix for the GroupedMatMul computation. The data type can be `torch.float16` or `torch.bfloat16`. This parameter must be 2D with shape `(BSK, H1)`. The data layout can be ND.
- **`gmm_weight`** (`Tensor`): Required. Right matrix for GroupedMatMul. The data type must be identical to that of `gmm_x`. This parameter must be 3D with shape `(e, H1, N1)`. The data layout can be ND.
- **`hcom`** (`str`): Required. Communicator name for expert parallelism. The string length must fall within the range of (0, 128).
- **`ep_world_size`** (`int`): Required. Size of the EP communication domain. Valid values: `8`, `16`, `32`, `64`, or `128`.

    <!-- npu="A3" id3 -->
    - <term>Atlas A3 products</term>: the supported values are 8, 16, 32, 64, and 128.
    <!-- end id3 -->
    <!-- npu="950" id4 -->
    - <term>Ascend 950DT products</term>: the supported values are 2, 4, 8, 16, 32, and 64.
    <!-- end id4 -->

- **`send_counts`** (`List[int]`): Required. Number of tokens sent to other devices. The data type can be `int`. The list length is `e * ep_world_size`, and the maximum value is `256`.
- **`recv_counts`** (`List[int]`): Required. Number of tokens received from other devices. The data type can be `int`. The list length is `e * ep_world_size`, and the maximum value is 256.
- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`send_counts_tensor`** (`Tensor`): Optional. The data type can be `torch.int`. The shape of this parameter is `(e * ep_world_size,)`. The data layout can be ND. **This parameter is not supported in the current version**. Retain the default value.
- **`recv_counts_tensor`** (`Tensor`): Optional. The data type can be `torch.int`. The shape of this parameter is `(e * ep_world_size,)`. The data layout can be ND. **This parameter is not supported in the current version**. Retain the default value.
- **`mm_x`** (`Tensor`): Optional. Left matrix for shared expert MatMul. This parameter must be provided when shared expert computation is fused. The data type can be `torch.float16` or `torch.bfloat16`. This parameter must be a 2D tensor, with shape `(BS, H2)`.
- **`mm_weight`** (`Tensor`): Optional. Right matrix for shared expert MatMul. This parameter must be provided when shared expert computation is fused. The data type must be identical to that of `mm_x`. This parameter must be a 2D tensor, with shape `(H2, N2)`.
- **`trans_gmm_weight`** (`bool`): Optional. Specifies whether to transpose the right matrix of GroupedMatMul. `True` enables transposition and `False` disables it (the default value).
- **`trans_mm_weight`** (`bool`): Optional. Specifies whether to transpose the right matrix of shared expert MatMul. `True` enables transposition and `False` disables it (the default value).
- **`permute_out_flag`** (`bool`): Optional. Specifies whether to output the result of the Permute operation. Valid values are `True` (outputs the Permute result) or `False` (does not output it, the default value).
- **`comm_mode`** (`str`): **Optional**. Communication engine mode. The default value is `None`.

    <!-- npu="A3" id5 -->
    - <term>Atlas A3 products</term>: the supported values are `None` and `ai_cpu`. If any value is passed, AI CPU communication is used.
    <!-- end id5 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950DT products</term>: the supported values are `None`, `ai_cpu`, and `ccu`. When the value is `None`, AI CPU communication is used.
    <!-- end id6 -->

## Return Values

- **`gmm_y`** (`Tensor`): Computation output, representing the final result of GroupedMatMul. The data type is identical to that of `gmm_x`. This parameter must be 2D with shape `(A, N1)`.
- **`mm_y`** (`Tensor`): Computation output of the shared expert MatMul. The data type is identical to `mm_x`. This parameter must be 2D with shape `(BS, N2)`. This output is generated only when `mm_x` and `mm_weight` are provided.
- **`permute_out`** (`Tensor`): Output of the Permute operation. The data type is identical to that of `gmm_x`.

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
  - BSK: number of tokens sent by the current device (BS\*K=BSK), which is the accumulated sum of the send\_counts parameters. The value range is (0, 52428800).
  - H1: hidden layer size of routed experts. The value range is (0, 65536).
  - H2: hidden layer size of shared experts. The value range is (0, 12288].
  - e: number of experts on a single device. e<=32, and the maximum supported value of e\*ep\_world\_size is 256.
  - N1: head\_num of routed experts. The value range is (0, 65536).
  - N2: head\_num of shared experts. The value range is (0, 65536).
  - BS: batch sequence size.
  - K: number of topK experts selected. The value range of K is [2, 8].
  - A: number of tokens received by the current device, which is the accumulated sum of the recv\_counts parameters.
  - The accumulated sum of A on all devices in the EP communication domain is equal to the accumulated sum of BSK on all devices.

## Example

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
        # Static graph: dynamic=False; dynamic graph: dynamic=True
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

## References

| Term | Definition |
| --- | --- |
| **MoE** (mixture of experts) | A neural network architecture that replaces the feedforward layer with multiple "expert" subnetworks and dynamically selects some experts for computation for each input token through a gating mechanism, thus controlling the computation cost while expanding the model capacity. |
| **Routing Expert** | An expert dynamically selected by the gating mechanism in MoE. After each token is routed through TopK, only a few routing experts are activated for computation. |
| **Shared Expert** | A fixed expert that all tokens in MoE pass through. It does not participate in routing selection and is used to capture general knowledge. It is computed in parallel with the routing experts. |
| **token** | The basic data unit processed by the model. In NLP, it usually corresponds to a word or subword. In the MoE context, each token is assigned to several experts for computation after gating. |
| **TopK routing** | The gating mechanism selects the K experts with the highest scores for each input token for processing. The value range of K is usually [2, 8]. |
| **EP** (Expert Parallelism) | A parallel strategy that distributes different experts across different compute devices (ranks). Each device stores only the weights of some experts and sends tokens to the devices holding the target experts for computation through cross-device communication. |
| **EP communication domain** | The communication group consisting of all compute devices participating in expert parallelism. `ep_world_size` indicates the number of devices in this communication domain. |
| **AlltoAllv** | A collective communication operation in which each device sends different amounts of data to every other device and receives data from every other device at the same time. The difference from the standard AlltoAll is that the amount of data sent and received between devices can be unequal. |
| **Permute** | A data rearrangement operation. After MoE communication, tokens are rearranged by the expert dimension so that the tokens of the same expert are stored contiguously in memory, facilitating the subsequent batch computation. |
| **GroupedMatMul** (GMM) | Grouped matrix multiplication. Matrix multiplication is performed separately for multiple matrix pairs, and the shapes of the data in each group can differ. It is suitable for the scenario in MoE where different experts process different numbers of tokens. |
| **MatMul** | Standard matrix multiplication. |
| **hidden size** | Size of the hidden layer dimension, that is, the length of the feature vector. Routing experts and shared experts can have different hidden sizes (H1 and H2). |
| **head\_num** | Number of attention heads. In this operator, it is used to describe the output dimensions of experts (N1 and N2). |
| **batch sequence size (BS)** | Batch sequence length, that is, the number of tokens in the input sequence. |
| **Number of experts (e)** | Number of experts deployed on a single compute device. The total number of experts on all devices is `e * ep_world_size`. |
