# torch\_npu.npu\_all\_gather\_base\_mm

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>| √    |

## Function<a name="en-us_topic_0000001694916914_section14441124184110"></a>

- Description: Fuses `allgather` and `matmul` in Tensor Parallelism (TP) scenarios to enable pipelined overlapping of communication and computation.

- Formulas:
    $x1$ indicates the input `x1`.

    Basic scenario:
    $$
    output = allgather(x1) \mathbin{@} x2 + bias
    $$
    $$
    gather\_out = allgather(x1)
    $$
    Quantization scenario:
    $$
    output = (allgather(x1\_scale) * x2\_scale) * (allgather(x1)\mathbin{@} x2 + bias)
    $$
    $$
    gather\_out = allgather(x1)
    $$

    In the quantization scenario formula, the `allgather(x1_scale)` operation is handled internally by the underlying operator. You do not need to manually execute an allgather operation on `x1_scale`. You only need to directly pass the local `x1_scale` of the current rank (with shape `(m, 1)`) into the interface. The underlying operator automatically aggregates the `x1_scale` tensors from all ranks before the computation. `x2_scale` is a globally shared `perchannel` quantization parameter that is identical across all ranks, so it does not require an `allgather` operation.

> [!NOTE]    
> When using this API, ensure that the driver firmware package and CANN package are both version 8.0.RC2 or a matching later version. Otherwise, errors such as a BUS ERROR will be raised.

## Prototype

```python
torch_npu.npu_all_gather_base_mm(x1, x2, hcom, world_size, bias=None, x1_scale=None, x2_scale=None, gather_index=0, gather_output=True, comm_turn=0, output_dtype=None, comm_mode=None) -> tuple[Tensor, Tensor]
```

## Parameters

- **`x1`** (`Tensor`): Required. Left matrix in matrix multiplication. The data type can be `float16`, `bfloat16`, or `int8`. The data layout can be ND. This parameter must be 2D with shape `(m, k)`. The axis dimensions must satisfy the matmul input requirements, where the second axis matches the first axis of `x2`, and the value range of `k` is [256, 65535).
- **`x2`** (`Tensor`): Required. Right matrix in matrix multiplication. The data type must be identical to `x1`. The data layout can be ND or NZ. NZ is supported only when `comm_mode` is set to `'aiv'`. This parameter must be 2D with shape `(k, n)`. The axis dimensions must satisfy the matmul input requirements, where the first axis matches the second axis of `x1`, and the value range of `k` is [256, 65535).
- **`hcom`** (`string`): Required. Name of the communication domain handle, which is obtained by calling the `get_hccl_comm_name` API.
- **`world_size`** (`int`): Required. Total number of ranks within the communication domain.
    - Atlas A2 training products: Configurations of 2, 4, and 8 ranks are supported. All-mesh networking over HCCS links is supported, where each rank connects to all other ranks.
    - Atlas A3 training products/Atlas A3 inference products Configurations of 2, 4, 8, 16, and 32 ranks are supported. Double-ring networking over HCCS links is supported, where multiple ranks form a ring sequentially and each rank connects only to its adjacent left and right ranks.
- **`bias`** (`Tensor`): Optional. The data type can be `float16` or `bfloat16`. The data layout can be ND. The data type must be identical to `x1`. This parameter must be a 1D tensor, where the size must be identical to that of the 1st dimension of `output`. **The current version does not support non-zero bias inputs.**
- **`x1_scale`** (`Tensor`): Optional. Dequantization parameter for the left matrix. The data type can be `float32`. The data layout is ND. The shape of this parameter is `(m, 1)`. `pertoken` quantization is supported.
- **`x2_scale`** (`Tensor`): Optional. Dequantization parameter for the right matrix. The data type can be `float32` or `int64`. The data layout is ND. The shape of this parameter is `(1, n)`. `perchannel` quantization is supported. If an `int64` input is required, call `torch_npu.npu_trans_quant_param` in advance to obtain the `int64` `x2_scale`.
- **`gather_index`** (`int`): Optional. Target operand for the gather operation, where `0` indicates gathering `x1` and `1` indicates gathering `x2`. The default value is `0`. **The current version only supports an input value of 0.**
- **`gather_output`** (`bool`): Optional. Specifies whether to return the gathered output. The default value is `True`.
- **`comm_turn`** (`int`): Optional. Communication slicing granularity between ranks. The default value is `0`, indicating the default slicing method. **The current version only supports an input value of 0.**
- **`output_dtype`** (`ScalarType`): Optional. Data type of the first output tensor. This parameter can be specified as `bfloat16` or `float16` only in quantization scenarios where both `x1_scale` and `x2_scale` are `float32`. The default value is `bfloat16`.
- **`comm_mode`** (`string`): Optional. Communication mode. Valid values are `'ai_cpu'` or `'aiv'`. The `ai_cpu` mode supports only the basic scenario. The `aiv` mode supports both the basic scenario and the quantization scenario. The default value is `ai_cpu`.

## Return Values<a name="en-us_topic_0000001694916914_section15236153161410"></a>

- **`output`** (`Tensor`): The first output tensor, representing the result of `allgather` + `matmul`.
In basic scenarios, the data type is identical to `x1`.
In quantization scenarios, if the data type of `x2_scale` is `int64`, the output data type is `float16`. If both `x1_scale` and `x2_scale` are `float32`, the output data type is specified by `output_dtype`, and the default value is `bfloat16`.
- **`gather_out`** (`Tensor`): The second output tensor, representing the result of `allgather`. Whether this tensor is returned is controlled by the `gather_output` parameter. If `gather_output` is `False`, an empty tensor is returned.

## Constraints

- `x1` does not support transposed input. If `x2` is transposed, the size of its first dimension must match the last dimension of `x1`, satisfying the requirements of the MatMul operation.
- When `comm_mode` is `ai_cpu`:
     - This API can be used in training scenarios.
     - This API supports graph mode. 
     - Atlas A2 training products: Communication fusion operators (AllGatherMatmul, MatmulReduceScatter, and MatmulAllReduce) within a single model must share the identical communication domain.
- When `comm_mode` is `aiv`, this API is supported in both training and inference scenarios.

## Examples<a name="en-us_topic_0000001694916914_section14459801435"></a>

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
            output, gather_output = torch_npu.npu_all_gather_base_mm(input, weight, hcomm_info, world_size, gather_output=gather_output)
            return output, gather_output
    def define_model(model, graph_type):
        import torchair
        if graph_type == 1:  # Traditional graph input mode, static shape + online compilation
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=False)
        elif graph_type == 2:  # ACLNN graph input mode, dynamic shape + binary file.
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
