# torch_npu.npu_quant_matmul_all_to_all

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- **API function**: Fuses quantized Matmul computation, Permute (to ensure contiguous addresses after communication), and AlltoAll communication. Computation is performed before communication. The K-C quantization mode (pertoken-perchannel quantization) and the MX quantization mode (a special pergroup-pergroup quantization mode) are supported.

- **Formulas**: Assume that the shape of the input `x1` is `(BS, H1)`, the shape of the input `x2` is `(H1, H2)`, and rankSize is the number of NPUs.
  - K-C quantization mode:

    $$
    computeOut = (x1@x2 + bias) * x1Scale * x2Scale
    $$

    $$
    permutedOut = computeOut.view(BS, rankSize, H2/rankSize).permute(1, 0, 2)
    $$

    $$
    output = AlltoAll(permutedOut).view(rankSize*BS, H2/rankSize)
    $$

  - MX quantization mode:

    $$
    computeOut = \sum_{0}^{\frac{k}{blockSize}=32}(x1@x2*(x1Scale*x2Scale)) + bias
    $$

    $$
    permutedOut = computeOut.view(BS, rankSize, H2/rankSize).permute(1, 0, 2)
    $$

    $$
    output = AlltoAll(permutedOut).view(rankSize*BS, H2/rankSize)
    $$

## Prototype

```python
torch_npu.npu_quant_matmul_all_to_all(x1, x2, hcom, world_size, bias=None, x1_scale=None, x2_scale=None, common_scale=None, x1_offset=None, x2_offset=None, x1_quant_mode=None, x2_quant_mode=None, common_quant_mode=None, group_sizes=None, all2all_axes=None, comm_quant_dtype=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None, output_scale_dtype=None, comm_scale_dtype=None, y_dtype=None, comm_mode=None) -> Tensor
```

## Parameters

- **`x1`** (`Tensor`): Required. Left matrix input of the fused operator and the left matrix of the Matmul computation, corresponding to x1 in the formulas. The data type can be `torch.float8_e5m2`, `torch.float8_e4m3fn`, or `torch_npu.float4_e2m1fn_x2`. The number of dimensions can only be 2D, with shape `(BS, H1)`. The data layout can be ND. **Non-contiguous Tensors are not supported.**
- **`x2`** (`Tensor`): Required. Right matrix input of the fused operator and the right matrix of the Matmul computation, corresponding to x2 in the formulas. The data type can be `torch.float8_e5m2`, `torch.float8_e4m3fn`, or `torch_npu.float4_e2m1fn_x2`. The number of dimensions can only be 2D, with shape `(H1, H2)`. The data layout can be ND. Transposed non-contiguous Tensors are supported.
- **`hcom`** (`str`): Required. String that identifies the column group on the host side, that is, the communication domain name. It is obtained by using the get_hccl_comm_name API.
- **`world_size`** (`int`): Required. Total number of ranks in the communication domain, corresponding to rankSize in the formulas. The supported range is [2, 4, 8, 16]. CCU supports only intra-node UB-domain interconnection, while AICPU supports inter-node UB-domain interconnection.
- **`bias`** (`Tensor`): Optional. Bias accumulated after the matrix multiplication, corresponding to bias in the formulas. The data type can be `torch.float32`. The number of dimensions can only be 1D, with shape `(H2)`. The data layout can be ND.
- **`x1_scale`** (`Tensor`): Optional. Quantization coefficient of the left matrix, corresponding to x1Scale in the formulas. In the K-C quantization scenario, the data type can be `torch.float32`, the number of dimensions is 1D, the shape is `(BS)`, and the data layout can be ND. In the MX quantization scenario, the data type can be `torch.float8_e8m0fnu`, the number of dimensions is 3D, the shape is `(BS, ceil(H1/64), 2)`, and the data layout can be ND.
- **`x2_scale`** (`Tensor`): Optional. Quantization coefficient of the right matrix, corresponding to x2Scale in the formulas. In the K-C quantization scenario, the data type can be `torch.float32`, the number of dimensions is 1D, the shape is `(H2)`, and the data layout can be ND. In the MX quantization scenario, the data type can be `torch.float8_e8m0fnu`, the number of dimensions is 3D, the shape is `(H2, ceil(H1/64), 2)`, and the data layout can be ND.
- **`common_scale`** (`Tensor`): Optional. Quantization coefficient for low-bit communication. Reserved parameter. In the current version, pass the default value.
- **`x1_offset`** (`Tensor`): Optional. Quantization bias of the left matrix. Reserved parameter. In the current version, pass the default value.
- **`x2_offset`** (`Tensor`): Optional. Quantization bias of the right matrix. Reserved parameter. In the current version, pass the default value.
- **`x1_quant_mode`** (`int`): Optional. Quantization method of the left matrix. In the current version, the value can be 3 or 6, representing pertoken quantization and MX quantization respectively.
- **`x2_quant_mode`** (`int`): Optional. Quantization method of the right matrix. In the current version, the value can be 2 or 6, representing perchannel quantization and MX quantization respectively.
- **`common_quant_mode`** (`int`): Optional. Quantization method for low-bit communication. Reserved parameter. In the current version, pass the default value.
- **`group_sizes`** (`List[int]`): Optional. Quantization group size in the three directions of the Matmul computation. It must be configured in the MX quantization scenario. In the K-C quantization scenario, the default value can be used and the configuration does not take effect.
    - When it is not None, only a 3-element list is supported, in the form [group_m, group_n, group_k], representing the quantization grouping on the m, n, and k dimensions respectively. For example, group_m indicates that group_m values on the m dimension correspond to one quantization parameter.
    - When one or more of [group_m, group_n, group_k] are 0, the API resets the value based on the input shapes of `x1`, `x2`, `x1_scale`, and `x2_scale`. Calculation principle: if group_m=0, the quantization group value in the m direction is inferred by the API. The inference formula is group_m=m/scale_m (m must be divisible by scale_m), where m is the m in the shape of `x1` and scale_m is the m in the shape of pertoken_scale. The n and k directions are similar.
    - **In the MX quantization scenario, this value must be configured as [1, 1, 32]**, or ensure that the derived value is [1, 1, 32] when it can be derived.

- **`all2all_axes`** (`List[int]`): Optional. Direction of the AlltoAll and Permute data exchange. It can be empty or `[-1, -2]`, indicating that the Matmul result is converted from `(BS, H2)` to `(BS*rankSize, H2/rankSize)`.
- **`comm_quant_dtype`** (`int`): Optional. Quantization data type for low-bit communication. Reserved parameter. In the current version, pass the default value.
- **`x1_dtype`** (`int`): Optional. Data type of the input left matrix. For data types not natively supported by PyTorch (such as `torch_npu.float4_e2m1fn_x2`), the data type must be configured by using this parameter.
- **`x2_dtype`** (`int`): Optional. Data type of the input right matrix. For data types not natively supported by PyTorch (such as `torch_npu.float4_e2m1fn_x2`), the data type must be configured by using this parameter.
- **`x1_scale_dtype`** (`int`): Optional. Data type of the quantization coefficient of the input left matrix. In the MX quantization scenario, `torch.float8_e8m0fnu` can be passed to explicitly specify the logical data type of the quantization coefficient.
- **`x2_scale_dtype`** (`int`): Optional. Data type of the quantization coefficient of the input right matrix. In the MX quantization scenario, `torch.float8_e8m0fnu` can be passed to explicitly specify the logical data type of the quantization coefficient.
- **`output_scale_dtype`** (`int`): Optional. Data type of the output quantization coefficient. Reserved parameter. In the current version, pass the default value.
- **`comm_scale_dtype`** (`int`): Optional. Data type of the low-bit communication quantization coefficient. Reserved parameter. In the current version, pass the default value.
- **`y_dtype`** (`int`): Optional. Output data type. The value is 5 (`torch.float16`), 6 (`torch.float32`), or 15 (`torch.bfloat16`).
- **`comm_mode`** (`str`): Optional. Communication engine mode. The default value is `None`. The value can be `None`, `ai_cpu`, or `ccu`. When `None` is passed, `ai_cpu` communication is used.

## Return Values

`Tensor`

Final computation result, output in the formulas. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The number of dimensions is 2D, with shape `(BS*rankSize, H2/rankSize)`. The data layout can be ND. Non-contiguous Tensors are not supported. When `y_dtype` is passed and is not undefined, the data type is determined by `y_dtype`. When `y_dtype` is the default value None or is undefined, the data type defaults to `torch.float32`, indicating high precision.

## Constraints

- This API supports training and inference scenarios.
- This API supports single-operator mode and graph mode.
- Empty Tensors are not supported for any input or output parameter.
- The communication domain name `hcom` cannot be an empty string, and its length range is [1, 127].
- Variables used in the parameter shapes:
  - BS: Size of the first dimension of the input left matrix, indicating the number of input sequences. The value range of BS*rankSize is [1, 2147483647].
  - H1: Size of the second dimension of the input left matrix and the first dimension of the input right matrix, indicating the hidden layer dimension. The value range is [1, 65535].
  - H2: Size of the second dimension of the input right matrix, indicating the length of the output sequence. The value range is [2, 2147483647], and it must be divisible by rankSize.

- The relationship between the enum values of `x1_quant_mode`, `x2_quant_mode`, and `common_quant_mode` and the quantization modes is as follows:
  - 0: no quantization
  - 1: pertensor
  - 2: perchannel
  - 3: pertoken
  - 4: pergroup
  - 5: perblock
  - 6: MX quantization
  - 7: pertoken dynamic quantization

- Currently, only the combinations [3, 2] and [6, 6] of `x1_quant_mode` and `x2_quant_mode` are supported, representing K-C quantization and MX quantization respectively.
- In the MX quantization scenario, `x2` must be transposed (that is, the .t() method must be used), and the shape after transposition must be `(H1, H2)`.
- In the MX quantization scenario, when the inputs `x1` and `x2` are `torch_npu.float4_e2m1fn_x2`, ceil(H1/32) must be even, and H1 must be even.
- In the current scenario, `x1_scale` and `x2_scale` must be passed and cannot be None.
- In the MX quantization mode, when the data type of `x1_dtype` and `x2_dtype` is `torch_npu.float4_e2m1fn_x2`, they must be set to 296. In other scenarios, the default value can be used.
- In the MX quantization scenario, `x1_scale_dtype` and `x2_scale_dtype` must be set to `torch.float8_e8m0fnu`. In other scenarios, the default value can be used.
- The detailed data type constraints for each quantization mode are as follows.

    **Table 1** K-C quantization mode data type constraints

    | x1 | x2 | bias | x1_scale | x2_scale | output | x1_quant_mode | x2_quant_mode |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float16` | 3 | 2 |
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.bfloat16` | 3 | 2 |
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | 3 | 2 |
    | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float16` | 3 | 2 |
    | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.bfloat16` | 3 | 2 |
    | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | 3 | 2 |
    | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float16` | 3 | 2 |
    | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.bfloat16` | 3 | 2 |
    | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | 3 | 2 |
    | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float16` | 3 | 2 |
    | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.bfloat16` | 3 | 2 |
    | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | 3 | 2 |

    **Table 2** MX quantization mode data type constraints

    | x1 | x2 | bias | x1_scale | x2_scale | output | x1_quant_mode | x2_quant_mode |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.float16` | 6 | 6 |
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.bfloat16` | 6 | 6 |
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.float32` | 6 | 6 |
    | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.float16` | 6 | 6 |
    | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.bfloat16` | 6 | 6 |
    | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.float32` | 6 | 6 |
    | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.float16` | 6 | 6 |
    | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.bfloat16` | 6 | 6 |
    | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.float32` | 6 | 6 |
    | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.float16` | 6 | 6 |
    | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.bfloat16` | 6 | 6 |
    | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.float32` | 6 | 6 |
    | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.float16` | 6 | 6 |
    | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.bfloat16` | 6 | 6 |
    | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.float32` | 6 | 6 |

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import numpy as np
    from ml_dtypes import float8_e4m3fn, float8_e5m2

    def run_quant_matmul_all_to_all(rank, world_size, master_ip, master_port, x1_shape, x2_shape, bias_shape, x1_scale_shape, x2_scale_shape):
        torch_npu.npu.set_device(rank)
        init_method = "tcp://" + master_ip + ":" + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > "2.0.1":
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
        x1_np = np.random.uniform(-1, 1, x1_shape).astype(float8_e4m3fn)
        x1_tensor = torch.from_numpy(x1_np.astype(np.float32)).to(torch.float8_e4m3fn).npu()
        x2_np = np.random.uniform(-1, 1, x2_shape).astype(float8_e4m3fn)
        x2_tensor = torch.from_numpy(x2_np.astype(np.float32)).to(torch.float8_e4m3fn).npu()
        bias_tensor = torch.randn(bias_shape, dtype=torch.float32).npu()
        x1_scale_tensor = torch.randn(x1_scale_shape, dtype=torch.float32).npu()
        x2_scale_tensor = torch.randn(x2_scale_shape, dtype=torch.float32).npu()
        output = torch_npu.npu_quant_matmul_all_to_all(
            x1_tensor,
            x2_tensor,
            hcom_info,
            world_size,
            bias_tensor,
            x1_scale_tensor,
            x2_scale_tensor
        )
        print("output: ", output)

    if __name__ == "__main__":
        worksize = 2
        master_ip = "127.0.0.1"
        master_port = "50001"
        x1_shape = [1024, 256]
        x2_shape =  [256, 3072]
        bias_shape = [3072]
        x1_scale_shape = [1024]
        x2_scale_shape = [3072]

        mp.spawn(
            run_quant_matmul_all_to_all,
            args=(worksize, master_ip, master_port, x1_shape, x2_shape, bias_shape, x1_scale_shape, x2_scale_shape),
            nprocs=worksize,
        )
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import numpy as np
    import pdb
    from ml_dtypes import int4, bfloat16, float8_e4m3fn, float8_e5m2, float8_e8m0fnu, float4_e2m1fn
    config = CompilerConfig()
    config.debug.graph_dump.type = 'pbtxt'
    npu_backend = tng.get_npu_backend(compiler_config=None)
    DTYPE_MAP = {
            "fp16": np.float16,
            "bf16": bfloat16,
            "fp32": np.float32,
            "fp64": np.float64,
            "int64": np.int64,
            "uint64": np.int64,
            "int32": np.int32,
            "uint32": np.int32,
            "int16": np.int16,
            "uint16": np.int16,
            "int8": np.int8,
            "uint8": np.uint8,
            "int4": int4,
            "float8_e4m3fn": float8_e4m3fn,
            "float8_e5m2": float8_e5m2,
            "hif8": np.uint8,
            "fp8_e8m0": float8_e8m0fnu,
            "fp4_e2m1": float4_e2m1fn,
            "fp4_e1m2": float4_e2m1fn,
        }
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, hcom, world_size, bias, x1_scale, x2_scale):
            return torch_npu.npu_quant_matmul_all_to_all(x1=x1, x2=x2, hcom=hcom, world_size=world_size,
                                                        bias=bias, x1_scale=x1_scale, x2_scale=x2_scale,
                                                        common_scale=None,
                                                        x1_offset=None,
                                                        x2_offset=None,
                                                        x1_quant_mode=3,
                                                        x2_quant_mode=2,
                                                        common_quant_mode=0,
                                                        group_sizes=[0,0,0],
                                                        all2all_axes=None,
                                                        comm_quant_dtype=28,
                                                        x1_dtype=None,
                                                        x2_dtype=None,
                                                        x1_scale_dtype=None,
                                                        x2_scale_dtype=None,
                                                        output_scale_dtype=None,
                                                        comm_scale_dtype=None,
                                                        y_dtype=6)
    def create_tensor(tensor_flag, shape=(), dtype="", transpose=False):
        device = torch.device("npu")
        if tensor_flag:
            array = np.random.uniform(-1, 1, shape).astype(DTYPE_MAP[dtype])
            if array.dtype in ["float8_e4m3fn", "float8_e5m2", "bfloat16"]:
                tensor = torch.reshape(torch.from_numpy(array.astype(np.float32)), shape).to(getattr(torch, str(array.dtype))).to(device)
            elif array.dtype == "float8_e8m0fnu":
                tensor = torch.reshape(torch.from_numpy(array.view(np.uint8)), shape).to(device)
            else:
                tensor = torch.reshape(torch.from_numpy(array), shape).to(device)
            if transpose:
                tensor = torch.transpose(tensor, 0, 1)
        else:
            tensor = None
        return tensor
    def run_quant_matmul_allto_all_graph(rank, world_size, master_ip, master_port, x1_shape, x2_shape, bias_shape, x1_scale_shape, x2_scale_shape):
        # get hcom_info
        torch_npu.npu.set_device(rank)
        init_method = "tcp://" + master_ip + ":" + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > "2.0.1":
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
        # generate input tensor
        x1_tensor = create_tensor(1, [1024, 256], "float8_e4m3fn")
        x2_tensor = create_tensor(1, [256, 3072], "float8_e4m3fn")
        bias_tensor = torch.randn(bias_shape, dtype=torch.float32).npu()
        x1_scale_tensor = torch.randn(x1_scale_shape, dtype=torch.float32).npu()
        x2_scale_tensor = torch.randn(x2_scale_shape, dtype=torch.float32).npu()
        cpu_model = Model()
        model = torch.compile(cpu_model, backend=npu_backend, dynamic=False, fullgraph=True)
        output = model(
            x1=x1_tensor,
            x2=x2_tensor,
            hcom=hcom_info,
            world_size=world_size,
            bias=bias_tensor,
            x1_scale=x1_scale_tensor,
            x2_scale=x2_scale_tensor
        )
        print("print output: ", output)
    if __name__ == "__main__":
        world_size = 2
        master_ip = "127.0.0.1"
        master_port = "50001"
        x1_shape = [1024, 256]
        x2_shape = [256, 3072]
        bias_shape = [3072]
        x1_scale_shape = [1024]
        x2_scale_shape = [3072]
        mp.spawn(
            run_quant_matmul_allto_all_graph,
            args=(world_size, master_ip, master_port, x1_shape, x2_shape, bias_shape, x1_scale_shape, x2_scale_shape),
            nprocs=world_size,
        )
    ```
