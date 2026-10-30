# torch_npu.npu_all_gather_base_mm

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->

## Function

- **Function**: in the TP (Tensor Parallelism) scenario, fuses allgather and matmul to implement pipelined communication and computation.

- **Formulas**:
    `x1` (that is, `input`) indicates the left matrix of the Matmul computation, `x2` indicates the right matrix of the Matmul computation, and `bias` indicates the bias of the Matmul computation.

    Basic scenario:
    $$
    output = allgather(x1) \mathbin{@} x2 + bias
    $$
    $$
    gather\_out = allgather(x1)
    $$

    <!-- npu="A3,910b" id22 -->
    Quantization scenario (supported only by <term>Atlas A2 products</term> and <term>Atlas A3 products</term>):
    $$
    output = (allgather(x1\_scale) * x2\_scale) * (allgather(x1) \mathbin{@} x2 + bias)
    $$
    $$
    gather\_out = allgather(x1)
    $$

    In the quantization scenario formula, `allgather(x1_scale)` is completed inside the underlying operator, and the user does not need to manually perform the allgather operation on `x1_scale`. The user only needs to pass the local `x1_scale` of the current rank (with the shape (m, 1)) to the API, and the underlying operator automatically aggregates the `x1_scale` of all ranks before participating in the computation. `x2_scale` is a globally shared perchannel quantization parameter, which is the same on all ranks and does not require an allgather operation.
    <!-- end id22 -->

> [!NOTE]
> When using this API, ensure that the driver firmware package and CANN package are both version 8.0.RC2 or a matching later version. Otherwise, errors such as BUS ERROR will be raised.

## Prototype

```python
torch_npu.npu_all_gather_base_mm(input, x2, hcom, world_size, *, bias=None, x1_scale=None, x2_scale=None, gather_index=0, gather_output=True, comm_turn=0, output_dtype=None, comm_mode=None) -> (Tensor, Tensor)
```

## Parameters

- **`input`** (`Tensor`): **Required**. Left matrix in the matrix multiplication. The data type can be `torch.float16` or `torch.bfloat16`, the data layout can be $ND$, the input shape can be 2D in the form of (m, k) or (k, n), the axes meet the input parameter requirements of the matmul operator, the k axes are equal, and the value range of the k axis is [256, 65535).
- **`x2`** (`Tensor`): Required. Right matrix in matrix multiplication. The data type must be identical to `x1`. The data layout can be ND or NZ. NZ is supported only when `comm_mode` is set to `'aiv'`. This parameter must be 2D with shape `(k, n)`. The axis dimensions must satisfy the matmul input requirements, where the first axis matches the second axis of `x1`, and the value range of `k` is [256, 65535).

    <!-- npu="A3,910b" id4 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the data layout can be $ND$ or $NZ$. $NZ$ is supported only when `comm_mode` is `aiv`.
    <!-- end id4 -->
    <!-- npu="950" id5 -->
    - <term>Ascend 950PR&950DT products</term>: The data layout supports $ND$.
    <!-- end id5 -->

- **`hcom`** (`string`): Required. Name of the communication domain handle, which is obtained by calling the `get_hccl_comm_name` API.
- **`world_size`** (`int`): Required. Total number of ranks within the communication domain.

    <!-- npu="910b" id6 -->
    - <term>Atlas A2 products</term>: 2, 4, and 8 devices are supported, and all-mesh HCCS link networking is supported (each device is connected to every other device).
    <!-- end id6 -->
    <!-- npu="A3" id7 -->
    - <term>Atlas A3 products</term>: Configurations of 2, 4, 8, 16, and 32 ranks are supported. Double-ring networking over HCCS links is supported, where multiple ranks form a ring sequentially and each rank connects only to its adjacent left and right ranks.
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT products</term>: 2, 4, 8, 16, 32, and 64 devices are supported. CCU supports only intra-node UB domain interconnection, while the AICPU supports inter-node UB domain interconnection.
    <!-- end id8 -->

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`bias`** (`Tensor`): Optional. The data type can be `torch.float16` or `torch.bfloat16`. The data layout can be ND. The data type must be identical to `x1`. This parameter must be a 1D tensor, where the size must be identical to that of the 1st dimension of `output`. **The current version does not support non-zero bias inputs.**

    <!-- npu="A3,910b" id9 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: **the scenario where the `bias` input is non-zero is not supported in the current version.**
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT products</term>: the scenario where the `bias` input is non-zero is supported.
    <!-- end id10 -->

- **`x1_scale`** (`Tensor`): Optional. Dequantization parameter for the left matrix. The data type can be `torch.float32`. The data layout is ND. The shape of this parameter is `(m, 1)`. `pertoken` quantization is supported.

    <!-- npu="A3,910b" id11 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the data type can be `torch.float32`.
    <!-- end id11 -->
    <!-- npu="950" id12 -->
    - <term>Ascend 950PR&950DT products</term>: **this parameter is not supported currently.**
    <!-- end id12 -->

- **`x2_scale`** (`Tensor`): Optional. Dequantization parameter for the right matrix. The data type can be `torch.float32` or `torch.int64`. The data layout is ND. The shape of this parameter is `(1, n)`. `perchannel` quantization is supported. If an `torch.int64` input is required, call `torch_npu.npu_trans_quant_param` in advance to obtain the `torch.int64` `x2_scale`.

    <!-- npu="A3,910b" id13 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the data type can be `torch.float32` or `torch.int64`. To pass the `torch.int64` data type, call torch\_npu.npu\_trans\_quant\_param in advance to obtain an `x2_scale` of the `torch.int64` data type.
    <!-- end id13 -->
    <!-- npu="950" id14 -->
    - <term>Ascend 950PR&950DT products</term>: **this parameter is not supported currently.**
    <!-- end id14 -->

- **`gather_index`** (`int`): Optional. Target operand for the gather operation, where `0` indicates gathering `x1` and `1` indicates gathering `x2`. The default value is `0`. **The current version only supports an input value of 0.**
- **`gather_output`** (`bool`): Optional. Specifies whether to return the gathered output. The default value is `True`. When `comm_mode` is `ai_cpu`, `gather_output` can be `True` or `False`. When `comm_mode` is `aiv`, `gather_output` can only be `True`.
- **`comm_turn`** (`int`): Optional. Communication slicing granularity between ranks. The default value is `0`, indicating the default slicing method. **The current version only supports an input value of 0.**
- **`output_dtype`** (`ScalarType`): Optional. Data type of the first output tensor. This parameter can be specified as `torch.bfloat16` or `torch.float16` only in quantization scenarios where both `x1_scale` and `x2_scale` are `torch.float32`. The default value is `torch.bfloat16`.

    <!-- npu="950" id15 -->
    - <term>Ascend 950PR&950DT products</term>: **this parameter is not supported currently.**
    <!-- end id15 -->

- **`comm_mode`** (`str`): Optional. Communication mode. The default value is `None`.

    <!-- npu="A3,910b" id16 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: `ai_cpu` and `aiv` are supported, and `None` indicates AICPU communication. The `ai_cpu` mode supports only the basic scenario, and the `aiv` mode supports the basic and quantization scenarios.
    <!-- end id16 -->
    <!-- npu="950" id17 -->
    - <term>Ascend 950PR&950DT products</term>: `ai_cpu`, `ccu`, and `None` are supported. When the value is `None`, AICPU communication is used.
    <!-- end id17 -->

## Return Values

- **`output`** (`Tensor`): The first output tensor, representing the result of `allgather` + `matmul`.

    <!-- npu="A3,910b" id18 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: in the basic scenario, the data type is the same as that of `input`. In the quantization scenario, when `x2_scale` is of the `torch.int64` data type, the output data type is `torch.float16`; when both `x1_scale` and `x2_scale` are `torch.float32`, the output data type is specified by `output_dtype` and defaults to `torch.bfloat16`.
    <!-- end id18 -->
    <!-- npu="950" id19 -->
    - <term>Ascend 950PR&950DT products</term>: only the basic scenario is supported, and the output data type is the same as that of `input`.
    <!-- end id19 -->

- **`gather_out`** (`Tensor`): The second output tensor, representing the result of `allgather`. Whether this tensor is returned is controlled by the `gather_output` parameter. If `gather_output` is `False`, an empty tensor is returned.

## Constraints

- This API supports single-operator mode and TorchAir graph mode.
- For `input`, a transposed tensor is not supported as input. If `x2` is transposed and passed as input, the size of its first dimension must be the same as the last dimension of `input`, meeting the computation conditions of matmul.
- **Communication mode constraints**:
  - When `comm_mode` is `ai_cpu`:
      - This API can be used in training scenarios.
      - This API supports graph mode.

      <!-- npu="910b" id20 -->
      - <term>Atlas A2 products</term>: the communication-computation fusion operators in a model (AllGatherMatmul, MatmulReduceScatter, and MatmulAllReduce) support only the same communication domain.
      <!-- end id20 -->

  - When `comm_mode` is `aiv`, both training and inference scenarios can be used.

<!-- npu="950" id21 -->
- <term>Ascend 950PR&950DT products</term>: the total amount of allgather\(x1\) collective communication data must not exceed 63\*256MB. The total amount of collective communication data is calculated as: m \* k \* sizeof\(x1\_dtype\) \* number of devices. Because the shapes differ, the internal implementation of the operator may differ, and the actually supported total communication volume may be slightly less than this value.
<!-- end id21 -->

## Example

- Single-operator call

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    def run_all_gather_base_mm(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank)

        tensor_allgather_shape = x1_shape
        single_shape = [x1_shape[0] // world_size, x1_shape[1]]

        input_ = torch.randn(single_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        output, gather_out = torch_npu.npu_all_gather_base_mm(input_, weight, hcomm_info, world_size)

    if __name__ == "__main__":
        worksize = 8
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16

        mp.spawn(run_all_gather_base_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    class ALLGATHER_MM_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, input, weight, hcomm_info, world_size, gather_output):
            output, gather_output = torch_npu.npu_all_gather_base_mm(input, weight, hcomm_info, world_size,
                                                                     gather_output=gather_output)
            return output, gather_output
    def define_model(model, graph_type):
        import torchair
        if graph_type == 1:  # Traditional graph mode, static shape + online compilation scenario
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=False)
        elif graph_type == 2:  # ACLNN graph mode, dynamic shape + binary
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=True)
        else:
            print("Error type")
        return model
    def get_graph(input, weight, hcomm_info, world_size, gather_output):
        model = ALLGATHER_MM_GRAPH_Model()
        model = define_model(model, 2)
        model_output = model(input, weight, hcomm_info, world_size, gather_output=gather_output)
        output_npu = model_output[0]
        gather_output_npu = model_output[1]
        return output_npu, gather_output_npu
    def run_all_gather_base_mm(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank)
        single_shape = [x1_shape[0] // world_size, x1_shape[1]]
        input = torch.randn(single_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        is_gather_out = True
        output, gather_out = get_graph(input, weight, hcomm_info, world_size, is_gather_out)
        print("output:", output)
    if __name__ == "__main__":
        worksize = 8
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16
        mp.spawn(run_all_gather_base_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```
