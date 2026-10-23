# torch_npu.npu_quant_gmm_alltoallv

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->

## Function

- **API function**: Fuses routed-expert GroupedMatmul and AlltoAllv. Computation is performed before communication. It is also fused in parallel with shared-expert MatMul computation. The T-T quantization mode (pertensor-pertensor quantization) and the MX quantization mode (a special pergroup-pergroup quantization mode) are supported.

- **Formulas**:
    - Routed experts:

        $$
        \begin{aligned}
        &gmm\_y = (gmm\_x \times gmm\_x\_scale) \mathbin{@} (gmm\_weight \times gmm\_weight\_scale) \\
        &unpermute\_out = Unpermute(gmm\_y) \\
        &y = AlltoAllv(unpermute\_out)
        \end{aligned}
        $$

        - gmm\_x is the left matrix of the routed-expert GroupedMatMul computation.
        - gmm\_x\_scale is the quantization parameter of the routed-expert left matrix.
        - gmm\_weight is the right matrix of the routed-expert GroupedMatMul computation.
        - gmm\_weight\_scale is the quantization parameter of the routed-expert right matrix.
        - gmm\_y is the output of the routed-expert GroupedMatMul computation and is used for the subsequent Unpermute computation.
        - unpermute\_out is the output of the Unpermute computation on gmm\_y and is used as the input of AlltoAllv communication.
        - y is the output of AlltoAllv communication on unpermute\_out.

    - Shared experts:

        $$
        mm\_y = (mm\_x \times mm\_x\_scale) \mathbin{@} (mm\_weight \times mm\_weight\_scale)
        $$

        - mm\_x is the left matrix of the shared-expert MatMul computation.
        - mm\_x\_scale is the quantization parameter of the shared-expert left matrix.
        - mm\_weight is the right matrix of the shared-expert MatMul computation.
        - mm\_weight\_scale is the quantization parameter of the shared-expert right matrix.
        - mm\_y is the output of the shared-expert MatMul computation.

## Prototype

```python
torch_npu.npu_quant_gmm_alltoallv(gmm_x, gmm_weight, gmm_x_scale, gmm_weight_scale, hcom, ep_world_size, send_counts, recv_counts, gmm_y_dtype, *, send_counts_tensor=None, recv_counts_tensor=None, mm_x=None, mm_weight=None, mm_x_scale=None, mm_weight_scale=None, comm_quant_scale=None, gmm_x_quant_mode=None, gmm_weight_quant_mode=None, mm_x_quant_mode=None, mm_weight_quant_mode=None, comm_quant_mode=None, group_size=None, gmm_x_dtype=None, gmm_weight_dtype=None, gmm_x_scale_dtype=None, gmm_weight_scale_dtype=None, mm_x_dtype=None, mm_weight_dtype=None, mm_x_scale_dtype=None, mm_weight_scale_dtype=None, comm_quant_dtype=None, mm_y_dtype=None, comm_mode=None) -> (Tensor, Tensor)
```

## Parameters

- **`gmm_x`** (`Tensor`): Required. Left matrix Tensor of the GroupedMatmul computation. The data type can be `torch_npu.hifloat8`, `torch.float8_e5m2`, `torch.float8_e4m3fn`, or `torch_npu.float4_e2m1fn_x2`. 2D is supported, with shape `(A, H1)`. The data layout can be ND. When the data type is float4, the inner axis H1 must be even so that 8 bits can be converted to two float4 values.
- **`gmm_weight`** (`Tensor`): Required. Right matrix of the GroupedMatmul. The data type is the same as that of `gmm_x`. 3D is supported, with shape `(e, H1, N1)`. The data layout can be ND. In fully quantized scenarios, when both `gmm_x` and `gmm_weight` are float4 types, only inference scenarios are supported. In this case, H1 of the input `gmm_x` must be even. When `gmm_weight` is not transposed, the inner axis N1 must be even; when it is transposed, the inner axis H1 must be even, so that 8 bits can be converted to two float4 values.
- **`gmm_x_scale`** (`Tensor`): Required. Quantization scaling coefficient of the left matrix. The data type can be `torch.float32` or `torch.float8_e8m0fnu`. In the pertensor quantization scenario, 1D is supported with shape `(1,)`. In the MX quantization scenario, 3D is supported with shape `(A, ceil(H1/64), 2)`. The data layout is ND.
- **`gmm_weight_scale`** (`Tensor`): Required. Quantization parameter of the right matrix. The data type can be `torch.float32` or `torch.float8_e8m0fnu`. In the pertensor quantization scenario, 1D is supported with shape `(1,)`. In the MX quantization scenario, 4D is supported with shape `(e, ceil(H1/64), N1, 2)`. The data layout is ND.
- **`hcom`** (`str`): Required. Communication domain name for expert parallelism (EP). The string length must be in the range (0,128).
- **`ep_world_size`** (`int`): Required. Size of the EP communication domain. The value can be 2, 4, 8, 16, 32, 64, 128, or 256. CCU supports only intra-node UB-domain interconnection, while AICPU supports inter-node UB-domain interconnection.
- **`send_counts`** (`List[int]`): Required. List of the number of tokens sent to other NPUs. The data type can be `torch.int64`. The array size is e \* ep_world_size.
- **`recv_counts`** (`List[int]`): Required. List of the number of tokens received from other NPUs. The data type can be `torch.int64`. The array size is e \* ep_world_size.
- **`gmm_y_dtype`** (`int`): Required. Data type of the routed-expert GroupedMatmul output Tensor `gmm_y` (for example, `torch.float16`). The data type can be `torch.float16` or `torch.bfloat16`.
- **`*`**: Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.
- **`send_counts_tensor`** (`Tensor`): Optional. The default value is `None`. Currently, only None is supported.
- **`recv_counts_tensor`** (`Tensor`): Optional. The default value is `None`. Currently, only None is supported.
- **`mm_x`** (`Tensor`): Optional. The default value is `None`. Left matrix in the shared-expert MatMul computation. The data type is the same as that of `gmm_x`. 2D is supported, with shape `(BS, H2)`. The data layout is ND. When the data type is float4, the inner axis H2 must be even so that 8 bits can be converted to two float4 values.
- **`mm_weight`** (`Tensor`): Optional. The default value is `None`. Right matrix in the shared-expert MatMul computation. The data type is the same as that of `gmm_x`. 2D is supported, with shape `(H2, N2)`. The data layout is ND. In fully quantized scenarios, when both `mm_x` and `mm_weight` are float4 types, only inference scenarios are supported. In this case, H2 of the input `mm_x` must be even. When `mm_weight` is not transposed, the inner axis N2 must be even; when it is transposed, the inner axis H2 must be even, so that 8 bits can be converted to two float4 values.
- **`mm_x_scale`** (`Tensor`): Optional. The default value is `None`. Quantization parameter of the shared-expert MatMul left matrix. The data type is `torch.float32`. In the pertensor quantization scenario, 1D is supported with shape `(1,)`. In the MX quantization scenario, 3D is supported with shape `(BS, ceil(H2/64), 2)`. The data layout is ND.
- **`mm_weight_scale`** (`Tensor`): Optional. The default value is `None`. Quantization parameter of the shared-expert MatMul right matrix. The data type is `torch.float32`. In the pertensor quantization scenario, 1D is supported with shape `(1,)`. In the MX quantization scenario, 3D is supported with shape `(ceil(H2/64), N2, 2)`. The data layout is ND.
- **`comm_quant_scale`** (`Tensor`): Optional. The default value is `None`. Quantization parameter for low-bit communication. The data type is `torch.float32`, and the number of dimensions is 1D. It is not supported currently.
- **`gmm_x_quant_mode`** (`int`): Optional. The default value is `None`. Quantization mode of the left matrix. In the current version, the value can be 1 or 6, representing pertensor quantization and MX quantization respectively.
- **`gmm_weight_quant_mode`** (`int`): Optional. The default value is `None`. Quantization mode of the right matrix. In the current version, the value can be 1 or 6, representing pertensor quantization and MX quantization respectively.
- **`mm_x_quant_mode`** (`int`): Optional. The default value is `None`. Quantization mode of the shared-expert left matrix. In the current version, the value can be 1 or 6, representing pertensor quantization and MX quantization respectively.
- **`mm_weight_quant_mode`** (`int`): Optional. The default value is `None`. Quantization mode of the shared-expert right matrix. In the current version, the value can be 1 or 6, representing pertensor quantization and MX quantization respectively.
- **`comm_quant_mode`** (`int`): Optional. The default value is `None`. Quantization mode of low-bit communication. Currently, only 0 is supported.
- **`group_size`** (`List[int]`): Optional. The default value is [0, 0, 0]. It indicates how many values in `gmm_x`, `gmm_weight`, `mm_x`, and `mm_weight` along the corresponding dimension can be quantized by one value in `gmm_x_scale`, `gmm_weight_scale`, `mm_x_scale`, and `mm_weight_scale`. `group_size` is [groupSizeM, groupSizeN, groupSizeK]. groupSizeM, groupSizeN, and groupSizeK represent the number of values corresponding to one quantization coefficient in each dimension.
    - The value of `group_size` is valid only in the MX quantization scenario. In other scenarios, pass [0, 0, 0].

- **`gmm_x_dtype`** (`int`): Optional. The default value is `None`. Actual data type of the routed-expert left matrix `gmm_x`. For data types not natively supported by PyTorch (such as `torch_npu.hifloat8` or `torch_npu.float4_e2m1fn_x2`), this parameter must be specified.
- **`gmm_weight_dtype`** (`int`): Optional. The default value is `None`. Actual data type of the routed-expert right matrix `gmm_weight`. For data types not natively supported by PyTorch (such as `torch_npu.hifloat8` or `torch_npu.float4_e2m1fn_x2`), this parameter must be specified.
- **`gmm_x_scale_dtype`** (`int`): Optional. The default value is `None`. Actual data type of the routed-expert left matrix quantization coefficient `gmm_x_scale`. In the MX quantization scenario, this parameter must be set to `torch_npu.float8_e8m0fnu` to specify the logical data type of the quantization coefficient.
- **`gmm_weight_scale_dtype`** (`int`): Optional. The default value is `None`. Actual data type of the routed-expert right matrix quantization coefficient `gmm_weight_scale`. In the MX quantization scenario, this parameter must be set to `torch_npu.float8_e8m0fnu` to specify the logical data type of the quantization coefficient.
- **`mm_x_dtype`** (`int`): Optional. The default value is `None`. Data type of the shared-expert left matrix `mm_x`. For data types not natively supported by PyTorch (such as `torch_npu.hifloat8` or `torch_npu.float4_e2m1fn_x2`), this parameter must be specified.
- **`mm_weight_dtype`** (`int`): Optional. The default value is `None`. Data type of the shared-expert right matrix `mm_weight`. For data types not natively supported by PyTorch (such as `torch_npu.hifloat8` or `torch_npu.float4_e2m1fn_x2`), this parameter must be specified.
- **`mm_x_scale_dtype`** (`int`): Optional. The default value is `None`. Data type of the shared-expert left matrix quantization coefficient `mm_x_scale`. In the MX quantization scenario, this parameter must be set to `torch_npu.float8_e8m0fnu` to specify the logical data type of the quantization coefficient.
- **`mm_weight_scale_dtype`** (`int`): Optional. The default value is `None`. Data type of the shared-expert right matrix quantization coefficient `mm_weight_scale`. In the MX quantization scenario, this parameter must be set to `torch_npu.float8_e8m0fnu` to specify the logical data type of the quantization coefficient.
- **`comm_quant_dtype`** (`int`): Optional. The default value is `None`. Data type after low-bit communication quantization. It is not supported currently.
- **`mm_y_dtype`** (`int`): Optional. The default value is `None`. Data type of the shared-expert output Tensor `mm_y`. The data type can be `torch.float16` or `torch.bfloat16`.
- **`comm_mode`** (`str`): Optional. Communication engine mode. The default value is `None`. The value can be `None`, `ai_cpu`, or `ccu`. When it is `None`, AICPU communication is used.

## Return Values

- **`y`** (`Tensor`): Final computation result of the routed-expert GroupedMatmul. The data type is specified by `gmm_y_dtype`. 2D is supported, with shape `(BSK, N1)`. The data layout is ND.
- **`mm_y`** (`Tensor`): Output of the shared-expert MatMul. The data type is specified by `mm_y_dtype`. 2D is supported, with shape `(BS, N2)`. It is output only when `mm_x` and `mm_weight` are passed. The data layout is ND.

## Constraints

- This API supports training and inference scenarios.
<!-- npu="950" id2 -->
- Communication engine constraints:
    - <term>Ascend 950DT products</term>: CCU communication is supported.
<!-- end id2 -->

- This API supports single-operator mode and graph mode for the T-T quantization scenario.
- Variables used in the shapes:
    - BSK: Number of tokens received by the current NPU, which is the sum of the `recvCounts` parameter. The value range is (0, 52428800).
    - H1: Hidden layer size of routed experts. The value range is (0, 65536).
    - H2: Hidden layer size of shared experts. The value range is (0, 12288].
    - e: Number of experts on a single NPU. e<=32, and e \* epWorldSize supports a maximum of 256.
    - N1: Intermediate layer dimension of routed experts. The value range is (0, 65536).
    - N2: Intermediate layer dimension of shared experts. The value range is (0, 65536).
    - BS: Batch sequence size.
    - K: Number of selected TopK experts. The value range of K is [2,8].
    - A: Number of tokens sent by the current NPU, which is the sum of the `sendCounts` parameter.
    - The sum of the A parameters of all NPUs in the EP communication domain is equal to the sum of the BSK parameters on all NPUs.

- The relationship between the values of `gmm_x_quant_mode`, `gmm_weight_quant_mode`, `mm_x_quant_mode`, and `mm_weight_quant_mode` and the quantization modes is as follows:
    - 0: no quantization
    - 1: pertensor
    - 2: perchannel
    - 3: pertoken
    - 4: pergroup
    - 5: perblock
    - 6: MX quantization
    - 7: pertoken dynamic quantization

- Currently, only the combinations [1, 1] and [6, 6] of `gmm_x_quant_mode` and `gmm_weight_quant_mode` are supported, representing T-T quantization and MX quantization respectively.
- Currently, only the combinations [1, 1] and [6, 6] of `mm_x_quant_mode` and `mm_weight_quant_mode` are supported, representing T-T quantization and MX quantization respectively. The quantization combination must be consistent with the combination of `gmm_x_quant_mode` and `gmm_weight_quant_mode`.

- `group_size`:
    - When one or more of groupSizeM, groupSizeN, and groupSizeK are 0, they are reset according to the input shapes of `gmm_x_scale`, `gmm_weight_scale`, `mm_x_scale`, `mm_weight_scale`, `gmm_x`, `gmm_weight`, `mm_x`, and `mm_weight` for calculation.
    - Setting principle: If groupSizeM=0, the quantization group value in the m direction is inferred by the API. The inference formula is groupSizeM = m / scaleM (m must be divisible by scaleM), where m is the m direction value in `gmm_x` and `mm_x`, and scaleM is the m direction value in `gmm_x_scale` and `mm_x_scale`. If groupSizeK=0, the quantization group value in the k direction is inferred by the API. The inference formula is groupSizeK = k / scaleK (k must be divisible by scaleK), where k is the k direction value in `gmm_x` and `mm_x`, and scaleK is the k direction value in `gmm_x_scale` and `mm_x_scale`. If groupSizeN=0, the quantization group value in the n direction is inferred by the API. The inference formula is groupSizeN = n / scaleN (n must be divisible by scaleN), where n is the n direction value in `gmm_weight` and `mm_weight`, and scaleN is the n direction value in `gmm_weight_scale` and `mm_weight_scale`.
    - If the reset conditions are met, when `gmm_x_scale`, `mm_x_scale`, and `mm_weight_scale` are 3D, `gmm_weight_scale` is 4D, and all data types are `torch.float8_e8m0fnu`, the value combination of [groupSizeM, groupSizeN, groupSizeK] is inferred as [1, 1, 32].
    - When the output Tensors (y, mm_y) of an NPU are both empty Tensors, `torch.distributed.barrier()` must be explicitly called to ensure that the process on this NPU waits synchronously for other NPUs to complete communication and computation. If synchronization is not added, AlltoAllv communication will block due to process asynchrony.

- The detailed input and output data type constraints for each quantization mode are as follows.

    **Table 1** T-T quantization data type constraints

    | gmm_x | gmm_weight | gmm_x_scale | gmm_weight_scale | gmm_x_quant_mode/gmm_weight_quant_mode | gmm_y | mm_x | mm_weight | mm_x_scale | mm_weight_scale | mm_x_quant_mode/mm_weight_quant_mode | mm_y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | [1, 1] | `torch.float16` | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | [1, 1] | `torch.float16` |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | [1, 1] | `torch.bfloat16` | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | [1, 1] | `torch.bfloat16` |

    **Table 2** MX quantization data type constraints

    | gmm_x | gmm_weight | gmm_x_scale | gmm_weight_scale | gmm_x_quant_mode/gmm_weight_quant_mode | gmm_y | mm_x | mm_weight | mm_x_scale | mm_weight_scale | mm_x_quant_mode/mm_weight_quant_mode | mm_y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.float16` | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.float16` |
    | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.float16` | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.float16` |
    | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.float16` | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.float16` |
    | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.float16` | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.float16` |
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` |
    | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` |
    | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` |
    | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` |
    | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.float16` | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.float16` |
    | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | [6, 6] | `torch.bfloat16` |

## Examples

- Single-operator call example

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp

    def run_npu_quant_gmm_alltoallv(rank, world_size, master_ip, master_port):
        torch_npu.npu.set_device(rank)
        init_method = f"tcp://{master_ip}:{master_port}"
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)

        BS, K = 128, 2
        H1, N1 = 256, 256
        H2, N2 = 256, 128
        e = 2
        ep_world_size = world_size
        total_tokens = BS * K
        out_dtype = torch.float16
        gmm_x = torch.randint(0, 30, (total_tokens, H1), dtype=torch.uint8).npu()
        gmm_weight = torch.randint(0, 30, (e, H1, N1), dtype=torch.uint8).npu()
        gmm_x_scale = torch.tensor([0.5], dtype=torch.float32).npu()
        gmm_weight_scale = torch.tensor([0.3], dtype=torch.float32).npu()
        mm_x = torch.randint(0, 30, (BS, H2), dtype=torch.uint8).npu()
        mm_weight = torch.randint(0, 30, (H2, N2), dtype=torch.uint8).npu()
        mm_x_scale = torch.tensor([0.4], dtype=torch.float32).npu()
        mm_weight_scale = torch.tensor([0.2], dtype=torch.float32).npu()
        send_counts = [total_tokens // (e * ep_world_size)] * (e * ep_world_size)
        recv_counts = [total_tokens // (e * ep_world_size)] * (e * ep_world_size)

        y, mm_y = torch_npu.npu_quant_gmm_alltoallv(
            gmm_x=gmm_x,
            gmm_weight=gmm_weight,
            gmm_x_scale=gmm_x_scale,
            gmm_weight_scale=gmm_weight_scale,
            hcom=hcom_info,
            ep_world_size=ep_world_size,
            send_counts=send_counts,
            recv_counts=recv_counts,
            gmm_y_dtype=out_dtype,
            mm_x=mm_x,
            mm_weight=mm_weight,
            mm_x_scale=mm_x_scale,
            mm_weight_scale=mm_weight_scale,
            gmm_x_quant_mode=1,
            gmm_weight_quant_mode=1,
            mm_x_quant_mode=1,
            mm_weight_quant_mode=1,
            comm_quant_mode=0,
            gmm_x_dtype=torch_npu.hifloat8,
            gmm_weight_dtype=torch_npu.hifloat8,
            gmm_x_scale_dtype=torch.float32,
            gmm_weight_scale_dtype=torch.float32,
            mm_x_dtype=torch_npu.hifloat8,
            mm_weight_dtype=torch_npu.hifloat8,
            mm_x_scale_dtype=torch.float32,
            mm_weight_scale_dtype=torch.float32,
            mm_y_dtype=out_dtype,
        )

    if __name__ == "__main__":
        world_size = 2
        master_ip = '127.0.0.1'
        master_port = '50001'
        mp.spawn(run_npu_quant_gmm_alltoallv, args=(world_size, master_ip, master_port), nprocs=world_size, join=True)

    ```

- Graph mode call example

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torchair
    import numpy as np
    from en_dtypes import hifloat8
    class GMM_ALLTOALLV_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, _gmm_x, _gmm_weight, _gmm_x_scale, _gmm_weight_scale, _hcom, _ep_world_size,
                    _send_counts, _recv_counts, _gmm_y_dtype, _mm_y_dtype, _mm_x, _mm_weight, _mm_x_scale,
                    _mm_weight_scale, _comm_quant_scale, _gmm_x_quant_mode, _gmm_weight_quant_mode,
                    _mm_x_quant_mode, _mm_weight_quant_mode, _comm_quant_mode, _gmm_x_dtype, _gmm_weight_dtype,
                    _mm_x_dtype, _mm_weight_dtype, _gmm_weight_scale_dtype=None, _gmm_x_scale_dtype=None,
                    _mm_x_scale_dtype=None, _mm_weight_scale_dtype=None, _trans_gmm_weight=False,
                    _trans_mm_weight=False):
            if _trans_gmm_weight:
                _gmm_weight = torch.transpose(_gmm_weight, -2, -1)
            if _trans_mm_weight and _mm_weight is not None:
                _mm_weight = _mm_weight.t()
            gmm_y, mm_y = torch_npu.npu_quant_gmm_alltoallv(
                _gmm_x,
                _gmm_weight,
                _gmm_x_scale,
                _gmm_weight_scale,
                _hcom,
                _ep_world_size,
                _send_counts,
                _recv_counts,
                _gmm_y_dtype,
                send_counts_tensor=None,
                recv_counts_tensor=None,
                mm_x=_mm_x,
                mm_weight=_mm_weight,
                mm_x_scale=_mm_x_scale,
                mm_weight_scale=_mm_weight_scale,
                comm_quant_scale=_comm_quant_scale,
                gmm_x_quant_mode=_gmm_x_quant_mode,
                gmm_weight_quant_mode=_gmm_weight_quant_mode,
                mm_x_quant_mode=_mm_x_quant_mode,
                mm_weight_quant_mode=_mm_weight_quant_mode,
                comm_quant_mode=_comm_quant_mode,
                group_size=None,
                gmm_x_dtype=_gmm_x_dtype,
                gmm_weight_dtype=_gmm_weight_dtype,
                gmm_x_scale_dtype=_gmm_x_scale_dtype,
                gmm_weight_scale_dtype=_gmm_weight_scale_dtype,
                mm_x_dtype=_mm_x_dtype,
                mm_weight_dtype=_mm_weight_dtype,
                mm_x_scale_dtype=_mm_x_scale_dtype,
                mm_weight_scale_dtype=_mm_weight_scale_dtype,
                comm_quant_dtype=None,
                mm_y_dtype=_mm_y_dtype
            )
            return gmm_y, mm_y
    def run_npu_gmm_alltoallv(rank, ep_world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype,
                            gmm_x_scale, gmm_w_scale):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=ep_world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
        input_arr = np.random.uniform(1, -1, gmm_x).astype(hifloat8)
        weight_arr = np.random.uniform(1, -1, gmm_w).astype(hifloat8)
        input = torch.from_numpy(input_arr.view(np.uint8)).npu()
        weight = torch.from_numpy(weight_arr.view(np.uint8)).npu()
        input_scale = torch.randn(gmm_x_scale, dtype=torch.float32).npu()
        weight_scale = torch.randn(gmm_w_scale, dtype=torch.float32).npu()
        model = GMM_ALLTOALLV_GRAPH_Model()
        npu_backend = torchair.get_npu_backend(compiler_config=None)
        # Static graph: dynamic=False. Dynamic graph: dynamic=True.
        model = torch.compile(GMM_ALLTOALLV_GRAPH_Model(), backend=npu_backend, dynamic=False)
        print(model(_gmm_x=input,
                    _gmm_weight=weight,
                    _gmm_x_scale=input_scale,
                    _gmm_weight_scale=weight_scale,
                    _hcom=hcom_info,
                    _ep_world_size=ep_world_size,
                    _send_counts=send_counts,
                    _recv_counts=recv_counts,
                    _gmm_y_dtype=torch.float16,
                    _mm_y_dtype=None,
                    _mm_x=None,
                    _mm_weight=None,
                    _mm_x_scale=None,
                    _mm_weight_scale=None,
                    _comm_quant_scale=None,
                    _gmm_x_quant_mode=1,
                    _gmm_weight_quant_mode=1,
                    _mm_x_quant_mode=1,
                    _mm_weight_quant_mode=1,
                    _comm_quant_mode=0,
                    _gmm_x_dtype=dtype,
                    _gmm_weight_dtype=dtype,
                    _mm_x_dtype=None,
                    _mm_weight_dtype=None,
                    _gmm_weight_scale_dtype=torch.float32,
                    _gmm_x_scale_dtype=torch.float32,
                    _mm_x_scale_dtype=None,
                    _mm_weight_scale_dtype=None,
                    _trans_gmm_weight=False,
                    _trans_mm_weight=False))
    if __name__ == "__main__":
        epWorkSize = 2
        e = 4
        master_ip = '127.0.0.1'
        master_port = '50001'
        BS = 512
        K = 8
        gmm_x_shape = [BS*K, 2048]
        gmm_weight_shape = [e, 2048, 2048]
        send_counts = [512] * (e * epWorkSize)
        recv_counts = [512] * (e * epWorkSize)
        dtype = torch_npu.hifloat8
        scale_shape = [1]
        mp.spawn(run_npu_gmm_alltoallv, args=(epWorkSize, master_ip, master_port, gmm_x_shape, gmm_weight_shape, send_counts, recv_counts, dtype, scale_shape, scale_shape), nprocs=epWorkSize)
    ```
