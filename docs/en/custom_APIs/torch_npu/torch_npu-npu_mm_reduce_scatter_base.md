# torch_npu.npu_mm_reduce_scatter_base

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products</term>   | √  |

## Function

- Fuses matrix multiplication (MatMul) and reduce_scatter collective communication operations in tensor parallelism (TP) scenarios. The fused kernel implements pipelined parallelism between computation and communication. The following quantization modes are supported:
    - `perchannel` quantization: For weight tensors, an independent scale factor and zero point are maintained for each channel along the output channel dimension (`Output Channel`).
    - `pertoken` quantization: For activation tensors, independent quantization parameters are maintained for each token along the sequence length dimension (`Sequence Length / Token`).

- Calculation formulas:  
    $x_1$ represents the input `input`. The scenario is determined jointly by `comm_mode` and the data type of `x1`: when `comm_mode` is `ai_cpu`, the basic scenario always applies; when `comm_mode` is `aiv`, the basic scenario applies if `x1` is `float16` or `bfloat16`, and the quantization scenario applies if `x1` is `int8`.

    - Basic scenario:
        $$
        output = reducescatter(x_1 \mathbin{@} x_2 + bias)
        $$

    - Quantization scenario:
        $$
        output = reducescatter((x1\_scale * x2\_scale) * (x_1 \mathbin{@} x_2 + bias))
        $$
        Usage of scale parameters in the quantization scenario:
        - When only `x2_scale` is provided:
          $$
          output = reducescatter(x2\_scale * (x_1 \mathbin{@} x_2 + bias))
          $$
        - When both `x1_scale` and `x2_scale` are provided:
          $$
          output = reducescatter((x1\_scale * x2\_scale) * (x_1 \mathbin{@} x_2 + bias))
          $$
        - `x1_scale` is broadcast to shape `[m, 1]`, and `x2_scale` is broadcast to shape `[1, n]`.

> [!NOTE]
> When using this API, ensure that the driver firmware package and CANN package are both the matching version `8.0.RC2` or later. Otherwise, errors such as `BUS ERROR` may occur.

## Prototype

```python
torch_npu.npu_mm_reduce_scatter_base(input, x2, hcom, world_size, *, reduce_op='sum', bias=None, x1_scale=None, x2_scale=None, comm_turn=0, output_dtype=None, comm_mode=None) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. The data type can be `float16`, `bfloat16`, or `int8`. The data layout can be ND. This parameter must be 2D with shape `(m, k)`.
- **`x2`** (`Tensor`): Required. The data type must be identical to that of `input`. The data layout can be `ND` or `NZ`. `NZ` is supported only when `comm_mode` is set to `aiv`. This parameter must be 2D with shape `(k, n)`. The axes must satisfy the input requirements of the MatMul operator, with the `k` dimensions equal (that is, the last dimension of `input` must equal the first dimension of `x2`). The value of `k` must be in the range `[256, 65535)`, and the `m` dimension of `input` must be divisible by `world_size`.
- **`hcom`** (`str`): Required. Communicator handle name obtained by calling the `get_hccl_comm_name` API.
- **`world_size`** (`int`): Required. Total number of ranks within the communication domain.
    - Atlas A2 training products: Configurations of 2, 4, and 8 ranks are supported. All-mesh networking over HCCS links is supported, where each rank connects to all other ranks.
    - Atlas A3 training products/Atlas A3 inference products: Configurations of 2, 4, 8, 16, and 32 ranks are supported. Double-ring networking over HCCS links is supported, where multiple ranks form a ring sequentially and each rank connects only to its adjacent left and right ranks.

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`reduce_op`** (`str`): Optional. Type of the reduce operation. Only the default value `'sum'` is supported.
- **`bias`** (`Tensor`): Optional. The data type can be `float16` or `bfloat16`. The data layout can be ND. The data type must be identical to that of `input`. This parameter must be a 1D tensor, where the size must be identical to that of the 1st dimension of `output`. In the current version, non-zero `bias` inputs are not supported.
- **`x1_scale`** (`Tensor`): Optional. Dequantization parameter for the left matrix of the MatMul operation. The data type can be `float32`. The data layout can be ND. The shape of this parameter is `(m, 1)`. `pertoken` quantization is supported.
- **`x2_scale`** (`Tensor`): Optional. Dequantization parameter for the right matrix of the MatMul operation. The data type can be `float32` or `int64`. The data layout can be ND. The shape of this parameter is `(1, n)`. `perchannel` quantization is supported. If an `int64` input is required, call `torch_npu.npu_trans_quant_param` in advance to obtain the `int64` `x2_scale`.
- **`comm_turn`** (`int`): Optional. Communication splitting granularity between ranks. The default value is `0`, indicating the default splitting mode. Currently, only the value `0` is supported.
- **`output_dtype`** (`ScalarType`): Optional. Output data type. This parameter can be specified as `bfloat16` or `float16` only in quantization scenarios where both `x1_scale` and `x2_scale` are `float32`. The default value is `bfloat16`.
- **`comm_mode`** (`str`): Optional. Communication mode. Valid values are `ai_cpu` or `aiv`. The `ai_cpu` mode supports only the basic scenario. The `aiv` mode supports both the basic scenario and the quantization scenario. The default value is `ai_cpu`.

## Return Values

`Tensor`

The shape of the output tensor is `(m // world_size, n)`.
In the basic scenario, the output data type is identical to that of `input`.
In quantization scenarios, if the data type of `x2_scale` is `int64`, the output data type is `float16`. If both `x1_scale` and `x2_scale` are `float32`, the output data type is specified by `output_dtype`, and the default value is `bfloat16`.

## Constraints

- `input` does not support transposed input. If `x2` is transposed, the size of its first dimension must match the last dimension of `input`, satisfying the requirements of the MatMul operation.
- `world_size` must equal the total number of ranks in the actual communication domain, and the `m` dimension of `input` must be divisible by `world_size`.
- **Communication Mode Constraints**
  - When `comm_mode` is `ai_cpu`:
       - This API is supported only in training scenarios.
       - This API supports graph mode.
       - <term>Atlas A2 training products</term>: The communication domains of the communication-compute fusion operators (AllGatherMatmul, MatmulReduceScatter, and MatmulAllReduce) within a model must be the same.
  - When `comm_mode` is `aiv`, this API can be used in both training and inference scenarios.

- **`comm_mode` Support Matrix**

| Scenario | `comm_mode` | `input` Data Type | Supported |
| --- | --- | --- | --- |
| Basic scenario | `ai_cpu` | `float16` / `bfloat16` | Yes |
| Quantization scenario | `ai_cpu` | `int8` | No |
| Basic scenario | `aiv` | `float16` / `bfloat16` | Yes |
| Quantization scenario | `aiv` | `int8` | Yes |

- **Scale Parameter Combination Constraints**

| Scenario | `x1_scale` | `x2_scale` | Output Data Type |
| --- | --- | --- | --- |
| Basic scenario | Not provided | Not provided | Same as `input` |
| Quantization scenario | Not provided | `float32` | Specified by `output_dtype`; defaults to `bfloat16` |
| Quantization scenario | `float32` | `float32` | Specified by `output_dtype`; defaults to `bfloat16` |
| Quantization scenario | `float32` | `int64` | Specified by `output_dtype`; defaults to `bfloat16` |
| Quantization scenario | Not provided | `int64` | Specified by `output_dtype`; defaults to `bfloat16` |

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    def run_mm_reduce_scatter_base(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank)
    
        input_ = torch.randn(x1_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        output = torch_npu.npu_mm_reduce_scatter_base(input_, weight, hcomm_info, world_size)
    
    if __name__ == "__main__":
        worksize = 8
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16
    
        mp.spawn(run_mm_reduce_scatter_base, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    class MM_REDUCESCATTER_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, input, weight, hcomm_info, world_size, reduce_op):
            output = torch_npu.npu_mm_reduce_scatter_base(input, weight, hcomm_info, world_size,
                                                          reduce_op=reduce_op)
            return output
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
    def get_graph(input, weight, hcomm_info, world_size):
        model = MM_REDUCESCATTER_GRAPH_Model()
        model = define_model(model, 2)
        model_output = model(input, weight, hcomm_info, world_size, reduce_op="sum")
        return model_output
    def run_mm_reduce_scatter_base(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank)
        input = torch.randn(x1_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        output = get_graph(input, weight, hcomm_info, world_size)
        print("output:", output)
    if __name__ == "__main__":
        worksize = 8
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16
        mp.spawn(run_mm_reduce_scatter_base, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```
