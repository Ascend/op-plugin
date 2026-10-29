# torch_npu.npu_all_gather_quant_mm

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

- API function: In TP splitting scenarios, this API performs AllGather collective communication on the left MM matrix x1, performs MatMul on the result and the right matrix x2, supports dequantization in quantization scenarios, and can output both the AllGather result gatherOut and the maximum value of the MatMul result amaxOut. This API is a functional extension of `torch_npu.npu_all_gather_base_mm`:
  <!-- npu="950" id4 -->
  - <term>Ascend 950DT products</term> added support for the low-precision data types `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`/`torch_npu.float4_e2m1fn_x2` (pertensor, perblock, and MX quantization are supported, and MX quantization supports mxfp8 and mxfp4 scenarios).
  <!-- end id4 -->
  <!-- npu="A3,910b" id21 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term> add support for `torch.int8` (pertoken and perchannel quantization are supported).
  <!-- end id21 -->

- Formulas:

  - Scenario 1: When the data types of x1 and x2 are `torch.float16`/`torch.bfloat16`, AllGather is performed on the input x1, and MatMul is performed on x1 and x2:

    $$
    output=AllGather(x1)@x2 + bias
    $$

    $$
    gatherOut=AllGather(x1)
    $$

  - Scenario 2: When the data types of x1 and x2 are `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` in the pertensor scenario, or the data types of x1 and x2 are `torch.int8`/`torch_npu.int4` in the perchannel and pertoken scenarios, and amaxOut is not output, AllGather is performed on the input x1, MatMul is performed on x1 and x2, and then dequantization is performed:

    $$
    output=(x1Scale*x2Scale)*(AllGather(x1)@x2 + bias)
    $$

    $$
    gatherOut=AllGather(x1)
    $$

  - Scenario 3: When the data types of x1 and x2 are `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` in the perblock scenario, and amaxOut is not output, when x1 is (m, k), x2 is (k, n), x1Scale is (ceilDiv(m, 128), ceilDiv(k, 128)), and x2Scale is (ceilDiv(k, 128), ceilDiv(n, 128)), AllGather is performed on the input x1 and x1Scale, perblock quantized MatMul is performed on x1 and x2, and then dequantization is performed:

    $$
    output=\sum_{0}^{\left \lfloor \frac{k}{blockSize=128} \right \rfloor} (AllGather(x1)_{pr}@x2_{rq}*(AllGather(x1Scale)_{pr}*x2Scale_{rq}))
    $$

    $$
    gatherOut=AllGather(x1)
    $$

  - Scenario 4: When the data types of x1 and x2 are `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.float4_e2m1fn_x2` in the MX quantization scenario, x1 is (m, k), x2 is (k, n) in the non-transposed scenario and (n, k) in the transposed scenario, x1Scale is (m, ceilDiv(k, 64), 2), and x2Scale is (ceilDiv(k, 64), n, 2) in the non-transposed scenario and (n, ceilDiv(k, 64), 2) in the transposed scenario, AllGather is performed on the input x1 and x1Scale, MatMul is performed on x1 and x2, and then dequantization is performed:

    $$
    output=\sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (AllGather(x1)_{pr}@x2_{rq}*(AllGather(x1Scale)_{pr}*x2Scale_{rq}))
    $$

    $$
    gatherOut=AllGather(x1)
    $$

> [!NOTE]
>
> - The quantization method (pertensor, perchannel, pertoken, perblock, or MX) is implicitly determined by the shapes of x1_scale and x2_scale. The API does not provide an explicit quantization method parameter.
> <!-- npu="950" id22 -->
> - <term>Ascend 950DT products</term>: pertensor, perblock, and MX quantization are supported.
> <!-- end id22 -->
> <!-- npu="A3,910b" id23 -->
> - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: pertoken and perchannel quantization are supported.
> <!-- end id23 -->
> - For data types not natively supported by PyTorch (`torch_npu.int4`, `torch_npu.hifloat8`, `torch_npu.float4_e2m1fn_x2`, and so on), the actual data types must be specified by using the x1_dtype, x2_dtype, x1_scale_dtype, and x2_scale_dtype parameters.

## Prototype

```python
torch_npu.npu_all_gather_quant_mm(self, x2, hcom, world_size, *, bias=None, x1_scale=None, x2_scale=None, quant_scale=None, block_size=0, gather_index=0, gather_output=True, comm_turn=0, group_sizes=None, amax_output=False, y_dtype=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None, comm_mode=None) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`self`** (`Tensor`): Required. Left MM matrix, $x1$ in the formula. The shape is 2D `(m, k)`. Only the non-transposed scenario is supported. The data layout can be ND.

  <!-- npu="950" id5 -->
  - <term>Ascend 950DT products</term>: The data type can be `torch.float16`, `torch.bfloat16`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch_npu.hifloat8`, or `torch_npu.float4_e2m1fn_x2`.
  <!-- end id5 -->
  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, `torch.int8`, or `torch_npu.int4`.
  <!-- end id6 -->

- **`x2`** (`Tensor`): Required. Right MM matrix, $x2$ in the formula. The shape is 2D, `(n, k)` when transposed and `(k, n)` when not transposed. Both transposed and non-transposed scenarios are supported, and non-contiguous Tensors are supported only in the transposed scenario. The data layout can be ND. The supported data type range is the same as that of `self`, and the data types of x1 and x2 must be the same (`torch.float8_e4m3fn` and `torch.float8_e5m2` can be mixed).
- **`hcom`** (`str`): Required. Communication domain name. It can be obtained by using `group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)`, where `group` is a process group of `torch.distributed`.
- **`world_size`** (`int`): Required. Total number of ranks in the communication domain. It must be a power of 2.

  <!-- npu="950" id7 -->
  - <term>Ascend 950DT products</term>: 2, 4, 8, 16, 32, and 64 NPUs are supported.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: 2, 4, and 8 NPUs are supported.
  <!-- end id8 -->

- **`*`**: Position delimiter for positional and keyword parameters. Parameters before it are positional and must be passed in order. Parameters after it are keyword parameters and must be passed by using key-value pairs; if not assigned, their default values are used.
- **`bias`** (`Tensor`): Optional. Bias, $bias$ in the formula. The shape is 1D `(n,)`. The data layout can be ND. The default value is `None`.

  <!-- npu="950" id9 -->
  - <term>Ascend 950DT products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. When x1 is `torch.float16` or `torch.bfloat16`, the data type of bias must be the same as that of x1. When x1 is `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch_npu.hifloat8`, or `torch_npu.float4_e2m1fn_x2`, the data type of bias must be `torch.float32` in the pertensor and MX quantization scenarios, and only `None` is supported in the perblock scenario.
  <!-- end id9 -->
  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: In the current version, only `None` is supported.
  <!-- end id10 -->

- **`x1_scale`** (`Tensor`): Optional. Dequantization parameter of the left MM matrix. The default value is `None`. When the data types of x1 and x2 are `torch.float16`/`torch.bfloat16`, only `None` is supported.

  <!-- npu="950" id11 -->
  - <term>Ascend 950DT products</term>: In the pertensor scenario, the shape is `[1]`. In the perblock scenario, the shape is `(ceilDiv(m, 128), ceilDiv(k, 128))`. In these scenarios, the data type can be `torch.float32`. In the MX quantization scenario, the data type is `torch.float8_e8m0fnu` and the shape is `(m, ceilDiv(k, 64), 2)`.
  <!-- end id11 -->
  <!-- npu="A3,910b" id12 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The data type can be `torch.float32`. In the pertoken scenario, the shape is `(m, 1)`.
  <!-- end id12 -->

- **`x2_scale`** (`Tensor`): Optional. Dequantization parameter of the right MM matrix. The default value is `None`. When the data types of x1 and x2 are `torch.float16`/`torch.bfloat16`, only `None` is supported.

  <!-- npu="950" id13 -->
  - <term>Ascend 950DT products</term>: In the pertensor scenario, the shape is `[1]`. In the perblock scenario, the shape is `(ceilDiv(k, 128), ceilDiv(n, 128))`. In these scenarios, the data type can be `torch.float32`. In the MX quantization scenario, the data type is `torch.float8_e8m0fnu`. The shape is `(n, ceilDiv(k, 64), 2)` in the transposed scenario and `(ceilDiv(k, 64), n, 2)` in the non-transposed scenario. Only transposed and non-transposed scenarios are supported.
  <!-- end id13 -->
  <!-- npu="A3,910b" id14 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The data type can be `torch.float32` or `torch.int64` (`torch.int64` is supported only when x1 and x2 are `torch.int8` or the output data type is `torch.float16`). In the perchannel scenario, the shape is `(1, n)`.
  <!-- end id14 -->

- **`quant_scale`** (`Tensor`): Optional. Quantization parameter. The default value is `None`. In the current version, only `None` is supported.
- **`block_size`** (`int`): Optional. Indicates how many values on the M and N axes of the MM output matrix can be quantized for the corresponding direction. The default value is `0`. block_size is formed by concatenating blockSizeM, blockSizeN, and blockSizeK. Each value occupies 16 bits. The formula is block_size = blockSizeK | blockSizeN << 16 | blockSizeM << 32. The K axis is not involved in the MM output matrix, and blockSizeK is fixed to 0. In the current version, only blockSizeM=blockSizeN=0 is supported, that is, only 0 can be passed.
- **`gather_index`** (`int`): Optional. Identifies the gather target. The default value is `0`. `0` indicates that the target is x1, and `1` indicates that the target is x2. In the current version, only `0` is supported.
- **`gather_output`** (`bool`): Optional. Whether to output the AllGather result gatherOut. The default value is `True`. When it is `True`, gatherOut is output. When it is `False`, an empty Tensor is returned.
- **`comm_turn`** (`int`): Optional. Number of communication splits, that is, total data volume/single communication volume. The default value is `0`. In the current version, only `0` is supported.
- **`group_sizes`** (`List[int]`): Optional. Dequantization group size. The default value is `None`. The list length must be 3, and the elements are groupSizeM, groupSizeN, and groupSizeK in order. Each element ranges from `[0, 65535]`. The concatenation formula is group_size = groupSizeK | groupSizeN << 16 | groupSizeM << 32. When group_sizes is `None` or an empty list, group_size is 0.

  <!-- npu="A3,910b" id15 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: In the current version, only group_sizes=`None` is supported (group_size is 0).
  <!-- end id15 -->
  <!-- npu="950" id16 -->
  - <term>Ascend 950DT products</term>: The group_size value is valid only when both x1_scale and x2_scale are 2D or higher. In other scenarios, pass 0. When one or more of groupSizeM, groupSizeN, and groupSizeK are 0, the API derives them again based on the shapes of x1, x2, x1_scale, and x2_scale: groupSizeM = m / scaleM (must be divisible, where m is the first dimension of x1 and scaleM is the first dimension of x1Scale), groupSizeK = k / scaleK, and groupSizeN = n / scaleN. Common value combinations: when x1_scale and x2_scale are 2D `torch.float32`, the derived values are [128, 128, 128], and the corresponding group_size is 549764202624. When x1_scale and x2_scale are 3D `torch.float8_e8m0fnu`, the derived values are [1, 1, 32], and the corresponding group_size is 4295032864.
  <!-- end id16 -->

- **`amax_output`** (`bool`): Optional. Whether to output the maximum value amaxOut of the MatMul result. The default value is `False`. In the current version, only `False` is supported. In this case, an empty Tensor is returned for amax.
- **`y_dtype`** (`int`): Optional. Data type of the output y (for example, `torch.float16`). The default value is `None`. When x1 is `torch.float16`/`torch.bfloat16`, the data type of y is the same as that of x1. If y_dtype is specified, it must be the same as x1. When x1 is a low-precision type such as `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`/`torch_npu.float4_e2m1fn_x2`/`torch.int8`/`torch_npu.int4`, y_dtype must be specified. The supported data types are `torch.float16`, `torch.bfloat16`, and `torch.float32`.
- **`x1_dtype`** (`int`): Optional. Data type of x1 (self) that actually participates in the computation. The default value is `None`. When x1 is a data type not natively supported by PyTorch (such as `torch_npu.int4`, `torch_npu.hifloat8`, or `torch_npu.float4_e2m1fn_x2`), this parameter must be specified, and the itemsize of the data type must be the same as the itemsize of the x1 Tensor.
- **`x2_dtype`** (`int`): Optional. Data type of x2 that actually participates in the computation. The default value is `None`. The value rules are the same as those of x1_dtype.
- **`x1_scale_dtype`** (`int`): Optional. Actual data type of x1_scale. The default value is `None`. When x1_dtype is `torch_npu.float4_e2m1fn_x2`, it must be `torch.float8_e8m0fnu`.
- **`x2_scale_dtype`** (`int`): Optional. Actual data type of x2_scale. The default value is `None`. When x2_dtype is `torch_npu.float4_e2m1fn_x2`, it must be `torch.float8_e8m0fnu`.
- **`comm_mode`** (`str`): Optional. Communication mode. The default value is `None` (equivalent to `"ai_cpu"`).

  <!-- npu="950" id17 -->
  - <term>Ascend 950DT products</term>: `"ai_cpu"` and `"ccu"` are supported.
  <!-- end id17 -->
  <!-- npu="A3,910b" id18 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: Only `"aiv"` is supported.
  <!-- end id18 -->

## Return Values

- **`y`** (`Tensor`): Result of AllGather communication and MatMul, $output$ in the formula. The shape is 2D `(m*world_size, n)`, where world_size is the number of NPUs. When x1 is `torch.float16`/`torch.bfloat16`, the data type is the same as that of x1. When x1 is a low-precision data type, the data type is specified by y_dtype and can be `torch.float16`, `torch.bfloat16`, or `torch.float32`.
- **`gather_out`** (`Tensor`): Result of AllGather communication, $gatherOut$ in the formula. When gather_output is `True`, the shape is 2D `(m*world_size, k)` and the data type is the same as that of x1. When gather_output is `False`, an empty Tensor is returned.
- **`amax`** (`Tensor`): Maximum value of the MatMul result, $amaxOut$ in the formula. When amax_output is `True`, the shape is `[1]` and the data type is `torch.float32`. When amax_output is `False` (the only supported scenario in the current version), an empty Tensor is returned.

## Constraints

- This API supports inference and training scenarios.
- This API supports graph mode.
- The deterministic implementation is used by default.
- The input self must be 2D with shape `(m, k)`. The input x2 must be 2D with shape `(k, n)` when not transposed and `(n, k)` when transposed. The k axes must be equal, and the value range of the k axis is `[256, 65535)`. The values of m and n must not exceed 2147483647.
- self supports only the non-transposed scenario. x2 supports both transposed and non-transposed scenarios.
- The data types of x1 and x2 must be the same. When x1 and x2 are `torch.float8_e4m3fn`/`torch.float8_e5m2`, either of the two data types can be used.

<!-- npu="950" id19 -->
- <term>Ascend 950DT products</term>:
  - 2, 4, 8, 16, 32, and 64 NPUs are supported.
  - Empty Tensor scenarios are supported: m and n can be empty, but k cannot be empty, and the following conditions must be met: m is empty, k is not empty, and n is not empty; m is not empty, k is not empty, and n is empty; m is empty, k is not empty, and n is empty.
  - When x1 and x2 are `torch_npu.float4_e2m1fn_x2`, x2 supports transposed and non-transposed scenarios, while x1 supports only the non-transposed scenario. The k axis must be even, and when x2 is not transposed, the n axis must also be even.
  - When group_size is 549764202624, bias must be `None`.
  - When comm_mode is `"ccu"`, only intra-node UB-domain interconnection is supported. `"ai_cpu"` supports inter-node UB-domain interconnection. When the `"ccu"` communication engine is used, the total AllGather(x1) collective communication data volume in a single communication domain must not exceed 63*256MB. The total collective communication data volume is calculated as m \* k \* sizeof(x1_dtype) \* number of NPUs. Because shapes differ, the internal implementation of the operator may vary, and the actual supported total communication volume may be slightly smaller than this value.
<!-- end id19 -->
<!-- npu="A3,910b" id20 -->
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - 2, 4, and 8 NPUs are supported.
  - Only `None` is supported for bias.
  - Empty Tensors are not supported.
  - When x1 and x2 are `torch_npu.int4`, k and n must be even.
  - comm_mode supports only `"aiv"`.
  - The communication buffer must be greater than or equal to 200MB.
<!-- end id20 -->

## Examples

### Scenario 1: 8-NPU scenario where `m` on each NPU is `16`, and the total `m` after `AllGather` is `128`

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

    self_t = torch.randn(single_shape, dtype=dtype).npu()
    x2 = torch.randn(x2_shape, dtype=dtype).npu()
    output, gather_out, _ = torch_npu.npu_all_gather_quant_mm(self_t, x2, hcomm_info, world_size, comm_mode="aiv") # When running on A2/A3, comm_mode="aiv" must be added.
    print(output.shape, output.dtype)
    print(gather_out.shape, gather_out.dtype)

if __name__ == "__main__":
    worksize = 8
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [128, 512]
    x2_shape = [512, 256]
    dtype = torch.float16

    mp.spawn(run_all_gather_base_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
```

The output is as follows:

```text
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.float16
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.float16
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.float16
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.float16
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.float16
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.float16
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.float16
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.float16
```

<!-- npu="A3,910b" id24 -->
### Scenario 2: perchannel and pertoken scenario where the data types of `x1` and `x2` are `torch.int8` (<term>Atlas A3 products</term> and <term>Atlas A2 products</term>)

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
def run_npu_all_gather_quant_mm(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
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
    self_t = torch.randint(-128, 127, single_shape, dtype=dtype).npu()
    x2 = torch.randint(-128, 127, x2_shape, dtype=dtype).npu()
    # pertoken scenario: the shape of x1_scale is (m, 1). perchannel scenario: the shape of x2_scale is (1, n).
    x1_scale = torch.rand((single_shape[0], 1), dtype=torch.float32).npu()
    x2_scale = torch.rand((1, x2_shape[1]), dtype=torch.float32).npu()
    output, gather_out, amax = torch_npu.npu_all_gather_quant_mm(
        self_t, x2, hcomm_info, world_size, bias=None, x1_scale=x1_scale, x2_scale=x2_scale,
        gather_output=True, y_dtype=torch.float16, comm_mode="aiv")
    print(output.shape, output.dtype)  # (torch.Size([128, 256]), torch.float16)
    print(gather_out.shape, gather_out.dtype)  # (torch.Size([128, 512]), torch.int8)

if __name__ == "__main__":
    worksize = 8
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [128, 512]
    x2_shape = [512, 256]
    dtype = torch.int8
    mp.spawn(run_npu_all_gather_quant_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
```

The output is as follows:

```text
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.int8
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.int8
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.int8
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.int8
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.int8
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.int8
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.int8
torch.Size([128, 256]) torch.float16
torch.Size([128, 512]) torch.int8
```
<!-- end id24 -->

<!-- npu="950" id25 -->
### Scenario 3: perblock scenario where the data types of `x1` and `x2` are `torch.float8_e4m3fn` (only <term>Ascend 950DT products</term>)

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import math
def run_npu_all_gather_quant_mm(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
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
    m_rank, k = single_shape
    n = x2_shape[1]
    self_t = torch.randn(single_shape).to(dtype).npu()
    x2 = torch.randn(x2_shape).to(dtype).npu()
    # perblock scenario: blockSize is 128, x1_scale and x2_scale are both float32, with shapes (ceilDiv(m,128), ceilDiv(k,128)) and (ceilDiv(k,128), ceilDiv(n,128)).
    x1_scale = torch.rand((math.ceil(m_rank / 128), math.ceil(k / 128)), dtype=torch.float32).npu()
    x2_scale = torch.rand((math.ceil(k / 128), math.ceil(n / 128)), dtype=torch.float32).npu()
    output, gather_out, amax = torch_npu.npu_all_gather_quant_mm(
        self_t, x2, hcomm_info, world_size, bias=None, x1_scale=x1_scale, x2_scale=x2_scale,
        gather_output=True, y_dtype=torch.float16)
    print(output.shape, output.dtype)  # (torch.Size([256, 256]), torch.float16)
    print(gather_out.shape, gather_out.dtype)  # (torch.Size([256, 512]), torch.float8_e4m3fn)
    print(amax)  # When amax_output is False, an empty Tensor is returned.

if __name__ == "__main__":
    worksize = 2
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [256, 512]
    x2_shape = [512, 256]
    dtype = torch.float8_e4m3fn
    mp.spawn(run_npu_all_gather_quant_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
```

The output is as follows:

```text
torch.Size([256, 256]) torch.float16
torch.Size([256, 512]) torch.float8_e4m3fn
None
torch.Size([256, 256]) torch.float16
torch.Size([256, 512]) torch.float8_e4m3fn
None
```
<!-- end id25 -->

<!-- npu="950" id26 -->
### Scenario 4: MX quantization scenario where the data types of `x1` and `x2` are `torch.float8_e4m3fn` (only <term>Ascend 950DT products</term>)

```python
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
import math
def run_npu_all_gather_quant_mm(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
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
    m_rank, k = single_shape
    n = x2_shape[1]
    self_t = torch.randn(single_shape).to(dtype).npu()
    x2 = torch.randn(x2_shape).to(dtype).npu()
    # MX scenario: the scale data type is float8_e8m0fnu, and the shapes are (m, ceilDiv(k, 64), 2) and (ceilDiv(k, 64), n, 2).
    x1_scale = torch.rand((m_rank, math.ceil(k / 64), 2)).to(torch.float8_e8m0fnu).npu()
    x2_scale = torch.rand((math.ceil(k / 64), n, 2)).to(torch.float8_e8m0fnu).npu()
    output, gather_out, amax = torch_npu.npu_all_gather_quant_mm(
        self_t, x2, hcomm_info, world_size, bias=None, x1_scale=x1_scale, x2_scale=x2_scale,
        gather_output=True, y_dtype=torch.float16)
    print(output.shape, output.dtype)
    print(gather_out.shape, gather_out.dtype)

if __name__ == "__main__":
    worksize = 2
    master_ip = '127.0.0.1'
    master_port = '50001'
    x1_shape = [64, 256]
    x2_shape = [256, 128]  # Transposed input (n, k)
    dtype = torch.float8_e4m3fn
    mp.spawn(run_npu_all_gather_quant_mm, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
```

The output is as follows:

```text
torch.Size([64, 128]) torch.float16
torch.Size([64, 256]) torch.float8_e4m3fn
torch.Size([64, 128]) torch.float16
torch.Size([64, 256]) torch.float8_e4m3fn
```

<!-- end id26 -->
