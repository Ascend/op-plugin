# torch_npu.npu_quant_mm_reduce_scatter

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->

## Function

- API function: Fuses MatMul (matrix multiplication) and ReduceScatter collective communication. After performing matrix multiplication (with optional bias addition and dequantization) on the input parameters x1 (self) and x2, it splits the M axis by the number of NPUs and performs ReduceScatter communication. It is commonly used in distributed scenarios such as MoE (MC2).<br>
  This API is a functional extension of `torch_npu.npu_mm_reduce_scatter_base`. In addition to supporting `torch.float16`/`torch.bfloat16` for x1 and x2:
    <!-- npu="950" id4 -->
    - <term>Ascend 950DT products</term> added support for the low-precision data types `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`/`torch_npu.float4_e2m1fn_x2` (pertensor, perblock, and MX quantization are supported, and MX quantization supports mxfp8 and mxfp4 scenarios).
    <!-- end id4 -->
    <!-- npu="A3,910b" id23 -->
    - <term>Atlas A3 products</term> and <term>Atlas A2 products</term> add support for `torch.int8` (pertoken and perchannel quantization are supported).
    <!-- end id23 -->

- Formulas:
  - Scenario 1: When the data types of x1 and x2 are `torch.float16`/`torch.bfloat16`, matmul is performed on the inputs x1, x2, and bias, and then ReduceScatter communication is performed.

    $$
    output=ReduceScatter(x1@x2 + bias_{optional})
    $$

  - Scenario 2: When the data types of x1 and x2 are `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` in the pertensor scenario, or the data types of x1 and x2 are `torch.int8` in the perchannel and pertoken scenarios, and amax is not output, matmul and dequant are performed on the inputs x1 and x2, followed by ReduceScatter communication.

    $$
    output=ReduceScatter((x1Scale*x2Scale)*(x1@x2) + bias_{optional})
    $$

  - Scenario 3: When the data types of x1 and x2 are `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` in the perblock scenario, and amax is not output, and x1 has shape (m, k), x2 has shape (k, n), x1Scale has shape (ceildiv(m, 128), ceildiv(k, 128)), and x2Scale has shape (ceildiv(k, 128), ceildiv(n, 128)), matmul and dequant are performed on the inputs x1 and x2, followed by ReduceScatter communication.

    $$
    output=ReduceScatter(\sum_{0}^{ceildiv} (x1_{pr}@x2_{rq}*(x1Scale_{pr}*x2Scale_{rq})))
    $$

    where

    $$
    ceildiv = \left \lceil \frac{k}{blockSize=128} \right \rceil
    $$

  - Scenario 4: When the data types of x1 and x2 are `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.float4_e2m1fn_x2` in the MX quantization scenario, and amax is not output, and x1 has shape (m, k) and x1Scale has shape (m, ceildiv(k, 64), 2), matmul and dequant are performed on the inputs x1 and x2, followed by ReduceScatter communication. x2 supports both transposed and non-transposed scenarios: when x2 is transposed, the shape of x2 is (n, k) and the shape of x2Scale is (n, ceildiv(k, 64), 2); when x2 is not transposed, the shape of x2 is (k, n) and the shape of x2Scale is (ceildiv(k, 64), n, 2).

    $$
    output=ReduceScatter(\sum_{0}^{ceildiv} (x1_{pr}@x2_{rq}*(x1Scale_{pr}*x2Scale_{rq})))
    $$

    where

    $$
    ceildiv = \left \lceil \frac{k}{blockSize=32} \right \rceil
    $$

> [!NOTE]
>
> When `comm_mode` is not passed, the default value depends on the platform:
> <!-- npu="A3,910b" id24 -->
> The default value for <term>Atlas A3 products</term> and <term>Atlas A2 products</term> is "aiv".
> <!-- end id24 -->
> <!-- npu="950" id25 -->
> The default value for <term>Ascend 950DT products</term> is "ai_cpu".
> <!-- end id25 -->

## Prototype

```python
torch_npu.npu_quant_mm_reduce_scatter(self, x2, hcom, world_size, *, reduce_op='sum', bias=None,
                                      x1_scale=None, x2_scale=None, quant_scale=None, block_size=0,
                                      comm_turn=0, group_sizes=None, amax_output=False, y_dtype=None,
                                      x1_dtype=None, x2_dtype=None, x1_scale_dtype=None,
                                      x2_scale_dtype=None, comm_mode=None) -> (Tensor, Tensor)
```

## Parameters

- **`self`** (`Tensor`): Required. Left MM matrix, $x1$ in the formulas. In the current version, only 2D input is supported, with shape [m, k], and only the non-transposed scenario is supported. m must be an integer multiple of the number of NPUs (world_size). The value range of the k axis is [256, 65535). The data layout can be ND. Non-contiguous Tensors are not supported. The supported data types are as follows:

  <!-- npu="A3,910b" id5 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int8`.
  <!-- end id5 -->
  <!-- npu="950" id6 -->
  - <term>Ascend 950DT products</term>: Supports `torch.float16`, `torch.bfloat16`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch_npu.hifloat8`, and `torch_npu.float4_e2m1fn_x2`.
  <!-- end id6 -->

- **`x2`** (`Tensor`): Required. Right MM matrix, $x2$ in the formulas. In the current version, only 2D input is supported. Both transposed and non-transposed scenarios are supported. For details about the shape requirements, see Constraints. Only non-contiguous Tensors caused by transposition are supported. The supported data types are as follows:

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int8`. The data layout can be ND or FRACTAL_NZ.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950DT products</term>: Supports `torch.float16`, `torch.bfloat16`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch_npu.hifloat8`, and `torch_npu.float4_e2m1fn_x2`. The data layout can be ND only.
  <!-- end id8 -->

- **`hcom`** (`str`): Required. Communication domain name. It is obtained by using the get_hccl_comm_name API.
- **`world_size`** (`int`): Required. Number of NPUs (rank_size).

  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: 2, 4, and 8 NPUs are supported.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950DT products</term>: 2, 4, 8, 16, 32, and 64 NPUs are supported.
  <!-- end id10 -->

- **`*`**: Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.
- **`reduce_op`** (`str`): Optional. Reduce operation type. In the current version, only "sum" is supported. The default value is 'sum'.
- **`bias`** (`Tensor`): Optional. $bias$ in the formulas. In the current version, only 1D input is supported, with shape (n,). None can be passed. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`.

  <!-- npu="A3,910b" id11 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: When the data types of x1 and x2 are `torch.int8`, the data type of bias can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. When the data type of x1 and x2 is `torch.float16`, the data type of bias must be `torch.float16` or `torch.float32`. When the data type of x1 and x2 is `torch.bfloat16`, the data type of bias must be `torch.bfloat16` or `torch.float32`.
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950DT products</term>: If the data type of x1 is `torch.float16` or `torch.bfloat16`, the data type of bias must be `torch.float16` or `torch.bfloat16`. If the data type of x1 is `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch_npu.hifloat8`, or `torch_npu.float4_e2m1fn_x2`, the data type of bias must be `torch.float32` in the pertensor and MX quantization scenarios. In the perblock scenario, only None is supported.
  <!-- end id12 -->

- **`x1_scale`** (`Tensor`): Optional. Dequantization parameter of the left MM matrix. None can be passed. When the data types of x1 and x2 are `torch.float16`/`torch.bfloat16`, only None is supported.

  <!-- npu="A3,910b" id13 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: In the pertoken scenario, the shape is (m, 1). The data type can be `torch.float32`.
  <!-- end id13 -->
  <!-- npu="950" id14 -->
  - <term>Ascend 950DT products</term>: In the pertensor scenario, the shape is [1]. In the perblock scenario, the shape is [ceildiv(m, 128), ceildiv(k, 128)]. The data type can be `torch.float32`. In the MX quantization scenario (MXFP8 and MXFP4), the data type is `torch.float8_e8m0fnu`, and the shape is (m, ceilDiv(k, 64), 2).
  <!-- end id14 -->

- **`x2_scale`** (`Tensor`): Optional. Dequantization parameter of the right MM matrix. None can be passed. When the data types of x1 and x2 are `torch.float16`/`torch.bfloat16`, only None is supported.

  <!-- npu="A3,910b" id15 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: In the perchannel scenario, the shape is (1, n). The data type can be `torch.float32` or `torch.int64` (supported only when the output data type is `torch.float16`).
  <!-- end id15 -->
  <!-- npu="950" id16 -->
  - <term>Ascend 950DT products</term>: In the pertensor scenario, the shape is [1]. In the perblock scenario, the shape is [ceildiv(k, 128), ceildiv(n, 128)]. The data type can be `torch.float32`. In the MX quantization scenario, the data type is `torch.float8_e8m0fnu`. The shape follows whether `x2` is transposed. For details about the shape requirements, see Constraints. Only non-contiguous Tensors caused by transposition are supported.
  <!-- end id16 -->

- **`quant_scale`** (`Tensor`): Optional. Quantization scale of the output matrix. Currently, only None is supported. The default value is None.
- **`block_size`** (`int`): Optional. Indicates how many values in the M and N axes of the MM output matrix can be quantized for the corresponding direction. It is formed by concatenating blockSizeM, blockSizeN, and blockSizeK. Each value occupies 16 bits. The formula is blockSize = blockSizeK | blockSizeN << 16 | blockSizeM << 32. The K axis is not involved in the MM output matrix, and blockSizeK is fixed to 0. In the current version, only blockSizeM=blockSizeN=0 is supported. The default value is 0.
- **`comm_turn`** (`int`): Optional. Number of communication splits, that is, total data volume/single communication volume. In the current version, only 0 is supported. The default value is 0.
- **`group_sizes`** (`List[int]`): Optional. Indicates how many values in `x1`/`x2` in the corresponding dimension can be dequantized by one value in `x1_scale`/`x2_scale`. The length is 3, in the order [groupSizeM, groupSizeN, groupSizeK]. Each value ranges from [0, 65535]. The internal concatenation formula is groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32. Passing None or all 0s indicates the default value. The default value is None.

  <!-- npu="950" id17 -->
  - <term>Ascend 950DT products</term>: The value of groupSize is valid only when both x1Scale and x2Scale inputs are 2D or higher. In other scenarios, pass 0. When one or more of groupSizeM/groupSizeN/groupSizeK are 0, they are inferred again based on the input shapes of x1/x2/x1Scale/x2Scale: groupSizeM = m / scaleM (m must be divisible by scaleM), groupSizeK = k / scaleK, and groupSizeN = n / scaleN. Generally, when x1Scale and x2Scale inputs are both 2D and their data types are both `torch.float32`, the value combination [groupSizeM, groupSizeN, groupSizeK] is inferred as [128, 128, 128]. When x1Scale and x2Scale inputs are both 3D and their data types are both `torch.float8_e8m0fnu`, it is inferred as [1, 1, 32].
  <!-- end id17 -->
  <!-- npu="A3,910b" id18 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: In the current version, only 0 is supported.
  <!-- end id18 -->

- **`amax_output`** (`bool`): Optional. Whether to output the maximum value of the MM computation. When True, the maximum value amax of the MM computation is returned. The default value is False.
- **`y_dtype`** (`int`): Optional. Output data type (ScalarType enum value). When the input is `torch.float16` or `torch.bfloat16`, the output should be consistent with the input data type (it can be omitted). When the input is another data type, it must be passed. The value can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The default value is None.
- **`x1_dtype`** (`int`): Optional. Data type of x1 (ScalarType enum value), used to interpret the storage type of x1 as a specified low-precision type. In the MX quantization scenario (MXFP4), it must be passed, and the value must be `torch_npu.float4_e2m1fn_x2`. The default value is None.
- **`x2_dtype`** (`int`): Optional. Data type of x2 (ScalarType enum value), used to interpret the storage type of x2 as a specified low-precision type. In the MX quantization scenario (MXFP4), it must be passed, and the value must be `torch_npu.float4_e2m1fn_x2`. The default value is None.
- **`x1_scale_dtype`** (`int`): Optional. Data type of x1_scale (TorchNPU data type enum value). In the MX quantization scenario (MXFP4), it must be passed, and the value must be `torch_npu.float8_e8m0fnu`. The default value is None.
- **`x2_scale_dtype`** (`int`): Optional. Data type of x2_scale (TorchNPU data type enum value). In the MX quantization scenario (MXFP4), it must be passed, and the value must be `torch_npu.float8_e8m0fnu`. The default value is None.
- **`comm_mode`** (`str`): Optional. Communication mode.

  <!-- npu="A3,910b" id19 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: In the current version, only "aiv" is supported. The default value is "aiv".
  <!-- end id19 -->
  <!-- npu="950" id20 -->
  - <term>Ascend 950DT products</term>: In the current version, "ai_cpu" or "ccu" is supported. The default value is "ai_cpu".
  <!-- end id20 -->

## Return Values

- **`output`** (`Tensor`): Result of ReduceScatter communication and MatMul computation, $output$ in the formulas. The shape is (m / world_size, n), where world_size is the number of NPUs. When the type of x1 is `torch.float16` or `torch.bfloat16`, the output type is consistent with that of x1. When the type of x1 is `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch_npu.hifloat8`, or `torch_npu.float4_e2m1fn_x2`, the output data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The data layout can be ND. Empty Tensors are supported only when the output type is `torch.float16` or `torch.bfloat16`.
- **`amax`** (`Tensor`): Maximum value of the MM computation, $amaxOut$ in the formulas. When `amax_output` is True, a `torch.float32` Tensor with shape [1] is returned. When `amax_output` is False, an empty Tensor is returned.

## Constraints

- This API supports training and inference scenarios.
- Deterministic computation: The deterministic implementation is used by default.
- Input constraints (all products):
  - Only the x2 matrix supports transposed/non-transposed. The x1 matrix supports only the non-transposed scenario.
  - The input x1 is 2D with shape (m, k). m must be an integer multiple of the number of NPUs (rank_size), and the value range of the k axis is [256, 65535).
  - The input x2 must be 2D with shape (k, n). The axes must meet the input requirements of the MM operator, and the k axes must be equal.
  - bias is 1D with shape (n,).
  - The output is 2D with shape (m / rank_size, n), where rank_size is the number of NPUs.

<!-- npu="950" id21 -->
- <term>Ascend 950DT products</term> constraints:
  - Communication constraints: In the current version, only comm_mode `"ai_cpu"` or `"ccu"` is supported. CCU communication and AICPU communication are supported. CCU supports only intra-node UB-domain interconnection, while AICPU supports inter-node UB-domain interconnection.
  - When the data types of x1 and x2 are `torch.float16`/`torch.bfloat16`, x1/x2 support empty Tensor scenarios. m and n can be empty, but k cannot be empty. When the data types of x1 and x2 are `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`/`torch_npu.float4_e2m1fn_x2`, empty Tensors are not supported.
  - When the data types of x1 and x2 are `torch.float16`/`torch.bfloat16`/`torch_npu.hifloat8`/`torch_npu.float4_e2m1fn_x2`, the data types of x1 and x2 must be the same. When the data types of x1 and x2 are `torch.float8_e4m3fn`/`torch.float8_e5m2`, either of the two can be used.
  - The input x2 supports transposed and non-transposed scenarios, but the input constraint requires its shape to be (k, n). This is to meet the input requirements of the MM operator, that is, the shape of the left matrix x1 should be (output row dimension, reduction axis), and the shape of the right matrix x2 should be (reduction axis, output column dimension), where the output row dimension is m, the reduction axis is k, and the output column dimension is n. Supporting the transposed scenario means that x2 can be constructed as a contiguous Tensor with shape (n, k), and x2.t() is executed before this API is called to obtain x2 with a view of (k, n) that meets the input requirements of the MM operator. In this case, x2 is non-contiguous, and x2 also supports non-contiguous Tensors only in this scenario. In the non-transposed scenario, the shape of x2 is (k, n). In this case, x2 is contiguous and can be directly used as input.
  - In the MX quantization scenario, the input x2_scale is 3D, and its shape follows whether x2 is transposed. When x2 is transposed, the shape of x2_scale is (n, ceildiv(k, 64), 2). When x2 is not transposed, the shape of x2_scale is (ceildiv(k, 64), n, 2).
  - In the MX quantization scenario, when the inputs x1 and x2 are `torch_npu.float4_e2m1fn_x2` (MXFP4 quantization), k must be even. When x2 is not transposed, n must also be even.
  - 2, 4, 8, 16, 32, and 64 NPUs are supported.
  - The total ReduceScatter collective communication data volume must not exceed 16 \* 256MB. The total collective communication data volume is calculated as m \* n \* sizeof(output_dtype). Because shapes differ, the internal implementation of the operator may vary, and the actual supported total communication volume may be slightly smaller than this value.
<!-- end id21 -->
<!-- npu="A3,910b" id22 -->
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term> constraints:
  - Only comm_mode `"aiv"` is supported, and the communication buffer must be greater than or equal to 200MB.
  - Empty Tensors are not supported.
  - The data types of x1 and x2 must be the same.
  - 2, 4, and 8 NPUs are supported.
<!-- end id22 -->
- The value of `world_size` must be in the range [2, 4, 8, 16, 32, 64].
- The K axes of the input self and x2 must be equal, and the M axis of self must be divisible by world_size.
- `group_sizes` must be a list of length 3, and each element ranges from [0, 65535].
- In the MX quantization scenario (MXFP4), `x1_dtype`, `x2_dtype`, `x1_scale_dtype`, and `x2_scale_dtype` are required. x1_dtype and x2_dtype must be `torch_npu.float4_e2m1fn_x2`, and x1_scale_dtype and x2_scale_dtype must be `torch_npu.float8_e8m0fnu`.

## Examples

### Scenario 1: Both `x1` and `x2` are `torch.float16` or `torch.bfloat16`

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
def run_npu_quant_mm_reduce_scatter(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)

    self_t = torch.randn(x1_shape, dtype=dtype).npu()
    x2 = torch.randn(x2_shape, dtype=dtype).npu()
    output, amax = torch_npu.npu_quant_mm_reduce_scatter(self_t, x2, hcomm_info, world_size, bias=None, x1_scale=None, x2_scale=None, amax_output=amax_output)
    print(output.shape, output.dtype)
    # When amax_output is False, an empty Tensor is returned.
    print(amax)  # torch.Size([0])

if __name__ == "__main__":
    worksize = 2
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [128, 512]
    x2_shape = [512, 64]
    dtype = torch.float16
    amax_output = False  # When amax_output is False, an empty Tensor is returned.
    mp.spawn(run_npu_quant_mm_reduce_scatter, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output), nprocs=worksize)
    amax_output = True  # When amax_output is True, a [1] float32 maximum value result is returned.
    mp.spawn(run_npu_quant_mm_reduce_scatter, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output), nprocs=worksize)
```

The output is as follows:

```text
torch.Size([64, 64]) torch.float16
None
torch.Size([64, 64]) torch.float16
None

torch.Size([64, 64]) torch.float16
torch.Size([64, 64]) torch.float16
tensor([0.], device='npu:0')
tensor([0.], device='npu:1')
```

### Scenario 2: perchannel and pertoken scenario where the data types of `x1` and `x2` are `torch.int8` (amax is not output)

> [!NOTE]
>
> Scenario 2 with `torch.int8` in the perchannel and pertoken scenarios (<term>Atlas A3 products</term> and <term>Atlas A2 products</term>) is similar: the data types of x1 and x2 are `torch.int8`. In the pertoken scenario, the shape of `x1_scale` is (m, 1), and `x2_scale` is not passed. In the perchannel scenario, the shape of `x2_scale` is (1, n), and `x1_scale` is not passed. The data type of the scale can be `torch.float32` (`x2_scale` in the perchannel scenario also supports `torch.int64`).

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
def run_npu_quant_mm_reduce_scatter(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)

    self_t = torch.randint(-128, 127, x1_shape, dtype=dtype).npu()
    x2 = torch.randint(-128, 127, x2_shape, dtype=dtype).npu()
    # pertoken scenario: the shape of x1_scale is (m, 1). perchannel scenario: the shape of x2_scale is (1, n).
    x1_scale = torch.rand((x1_shape[0], 1), dtype=torch.float32).npu()
    x2_scale = torch.rand((1, x2_shape[1]), dtype=torch.float32).npu()
    output, amax = torch_npu.npu_quant_mm_reduce_scatter(self_t, x2, hcomm_info, world_size,
                                                         bias=None, x1_scale=x1_scale, x2_scale=x2_scale,
                                                         amax_output=amax_output, y_dtype=torch.float16)
    print(output.shape, output.dtype) # (torch.Size([64, 64]), torch.float16)


if __name__ == "__main__":
    worksize = 2
    master_ip = '127.0.0.1'
    master_port = '50002'
    x1_shape = [128, 512]
    x2_shape = [512, 64]
    dtype = torch.int8
    amax_output = False  # amax is not output in scenario 2.
    mp.spawn(run_npu_quant_mm_reduce_scatter, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output), nprocs=worksize)
```

The output is as follows:

```text
torch.Size([64, 64]) torch.float16
torch.Size([64, 64]) torch.float16
```

### Scenario 3: perblock scenario where the data types of `x1` and `x2` are `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` (amax is not output)

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import math
def run_npu_quant_mm_reduce_scatter(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)

    m, k = x1_shape
    n = x2_shape[1]
    self_t = torch.randn(x1_shape).to(dtype).npu()
    x2 = torch.randn(x2_shape).to(dtype).npu()
    # perblock scenario: blockSize is 128, and the shapes of the scales are (ceildiv(m, 128), ceildiv(k, 128)) and (ceildiv(k, 128), ceildiv(n, 128)).
    x1_scale = torch.rand((math.ceil(m / 128), math.ceil(k / 128)), dtype=torch.float32).npu()
    x2_scale = torch.rand((math.ceil(k / 128), math.ceil(n / 128)), dtype=torch.float32).npu()
    output, amax = torch_npu.npu_quant_mm_reduce_scatter(self_t, x2, hcomm_info, world_size, bias=None,
                                                          x1_scale=x1_scale, x2_scale=x2_scale,
                                                          amax_output=amax_output, y_dtype=torch.float16)
    print(output.shape, output.dtype) # (torch.Size([64, 64]), torch.float16)
    # When amax_output is False, an empty Tensor is returned.
    print(amax)  # torch.Size([0])

if __name__ == "__main__":
    worksize = 2
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [128, 512]
    x2_shape = [512, 128]
    dtype = torch.float8_e4m3fn
    amax_output = False  # amax is not output in scenario 3.
    mp.spawn(run_npu_quant_mm_reduce_scatter, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output), nprocs=worksize)

```

The output is as follows:

```text
torch.Size([64, 128]) torch.float16
None
torch.Size([64, 128]) torch.float16
None
```

### Scenario 4: MX quantization scenario where the data types of `x1` and `x2` are `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.float4_e2m1fn_x2` (amax is not output, MXFP8 is used as an example)

> [!NOTE]
>
> In the MXFP4 case of scenario 4 (the data types of x1 and x2 are `torch_npu.float4_e2m1fn_x2`), `x1_dtype` and `x2_dtype` (value `torch_npu.float4_e2m1fn_x2`) and `x1_scale_dtype` and `x2_scale_dtype` (value `torch_npu.float8_e8m0fnu`) must also be passed, and k must be even.

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import math
def run_npu_quant_mm_reduce_scatter(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output):
    torch_npu.npu.set_device(rank)
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)

    m, k = x1_shape   # x1_shape is (m, k)
    n = x2_shape[1]   # MX quantization supports only transposed input of x2 and x2_scale. After transposition, the shape of x2 is (k, n).
    self_t = torch.randn(x1_shape).to(dtype).npu()
    x2 = torch.randn(x2_shape).to(dtype).npu()
    # MX quantization scenario: the scale data type is float8_e8m0fnu, and the shapes are (m, ceildiv(k, 64), 2) and (ceildiv(k, 64), n, 2).
    x1_scale = torch.rand((m, math.ceil(k / 64), 2)).to(torch.float8_e8m0fnu).npu()
    x2_scale = torch.rand((math.ceil(k / 64), n, 2)).to(torch.float8_e8m0fnu).npu()
    output, amax = torch_npu.npu_quant_mm_reduce_scatter(self_t, x2, hcomm_info, world_size, bias=None,
                                                          x1_scale=x1_scale, x2_scale=x2_scale,
                                                          amax_output=amax_output, y_dtype=torch.float16)
    print(output.shape, output.dtype) # (torch.Size([64, 64]), torch.float16)
    # When amax_output is False, an empty Tensor is returned.
    print(amax)  # torch.Size([0])

if __name__ == "__main__":
    worksize = 2
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [128, 256]
    x2_shape = [256, 64] 
    dtype = torch.float8_e4m3fn
    amax_output = False  # amax is not output in scenario 4.
    mp.spawn(run_npu_quant_mm_reduce_scatter, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype, amax_output), nprocs=worksize)
```

The output is as follows:

```text
torch.Size([64, 64]) torch.float16
None
torch.Size([64, 64]) torch.float16
None
```
