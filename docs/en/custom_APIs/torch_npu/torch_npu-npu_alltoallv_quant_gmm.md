# torch_npu.npu_alltoallv_quant_gmm

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product                                                         | Supported |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950DT</term>     | √  |

## Function

- Fuses `AlltoAllv` communication and `GroupedMatMul` for routed experts, where communication is executed before computation. The operation is also fused with shared expert `MatMul` computation to enable parallel execution.

- Formulas:

    Assume that the total number of devices in the communication domain is `epWorldSize` and each device has `e` routed experts after communication. `GroupedMatMul` on each device is responsible solely for computing local experts. The computation formulas for each device are defined as follows:

    - Shared expert MatMul computation on the local device:

        $$
        mm\_y = mm\_x\_scale \times mm\_weight\_scale \times (mm\_x \mathbin{@} mm\_weight)
        $$

        - `mm_y` represents the output of the shared expert `MatMul`.
        - `mm_x` represents the left matrix input of the shared expert `MatMul`.
        - `mm_x_scale` represents the quantization scale of the left matrix `mm_x` of the shared expert.
        - `mm_weight` represents the right matrix input of the shared expert `MatMul`.
        - `mm_weight_scale` represents the quantization scale of the right matrix `mm_weight` of the shared expert.

    - AlltoAllv communication and Permute:

        $$
        \begin{aligned}
        &permute\_out = Permute(AlltoAllv(gmm\_x))
        \end{aligned}
        $$

        - `permute_out` represents the output after the `Permute` operation.
        - `gmm_x` represents the original left matrix input of the routed experts on the local device prior to communication.

    - GroupedMatMul computation for local routed experts along the expert dimension:

        $$
        gmm\_y = gmm\_x\_scale \times gmm\_weight\_scale \times (permute\_out \mathbin{@} gmm\_weight)
        $$

        - `gmm_weight` represents the right matrix input of the `GroupedMatMul` for the routed experts on the local device.
        - `gmm_x_scale` represents the quantization scale of the routed experts' left matrix `gmm_x`.
        - `gmm_weight_scale` represents the quantization scale of the routed experts' right matrix `gmm_weight`.

## Prototype

```python
torch_npu.npu_alltoallv_quant_gmm(gmm_x, gmm_weight, gmm_x_scale, gmm_weight_scale, hcom, ep_world_size, send_counts, recv_counts, gmm_y_dtype, *, send_counts_tensor=None, recv_counts_tensor=None, mm_x=None, mm_weight=None, mm_x_scale=None, mm_weight_scale=None, gmm_x_quant_mode=None, gmm_weight_quant_mode=None, mm_x_quant_mode=None, mm_weight_quant_mode=None, permute_out_flag=False, group_size=None, gmm_x_dtype=None, gmm_weight_dtype=None, gmm_x_scale_dtype=None, gmm_weight_scale_dtype=None, mm_x_dtype=None, mm_weight_dtype=None, mm_x_scale_dtype=None, mm_weight_scale_dtype=None, mm_y_dtype=None, comm_mode=None) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`gmm_x`** (`Tensor`): Required. Original left matrix input of routed experts on the local device before communication. The data type can be `hifloat8`, `float8_e4m3fn`, `float8_e5m2`, or `float4_e2m1fn_x2`. This parameter must be a 2D tensor with shape $(BSK, H1)$. The data layout can be `ND`. When the data type is `float4_e2m1fn_x2`, inner dimension `H1` must be even and cannot be `2` to ensure that 8 bits can be converted into two `float4_e2m1fn_x2` elements.
- **`gmm_weight`** (`Tensor`): Required. Right matrix input of the `GroupedMatMul` for routed experts on the local device. The data type can be `hifloat8`, `float8_e4m3fn`, `float8_e5m2`, or `float4_e2m1fn_x2`. This parameter must be a 3D tensor with shape $(e, H1, N1)$. The data layout can be `ND`. In the MX quantization scenario, when both `gmm_x` and `gmm_weight` use the `float4_e2m1fn_x2` data type, only inference scenarios are supported. In this case, inner dimension `H1` of `gmm_x` must be even and cannot be `2`. When `gmm_weight` is not transposed, inner dimension `N1` must be even; when it is transposed, inner dimension `H1` must be even and cannot be `2`, to ensure that 8 bits can be converted into two `float4_e2m1fn_x2` elements.
- **`gmm_x_scale`** (`Tensor`): Required. Quantization scale of routed expert left matrix `gmm_x`. The data type can be `float32` or `float8_e8m0fnu`. In the per-tensor quantization scenario, this parameter must be a 1D tensor with shape $(1)$. In the MX quantization scenario, this parameter must be a 3D tensor with shape $(BSK, \operatorname{ceil}(H1/64), 2)$. The data layout can be `ND`.
- **`gmm_weight_scale`** (`Tensor`): Required. Quantization scale of routed expert right matrix `gmm_weight`. The data type can be `float32` or `float8_e8m0fnu`. In the per-tensor quantization scenario, this parameter must be a 1D tensor with shape $(1)$. In the MX quantization scenario, this parameter must be a 4D tensor with shape $(e, \operatorname{ceil}(H1/64), N1, 2)$. The data layout can be `ND`.
- **`hcom`** (`str`): Required. String identifying the communication group on the host side, that is, the name of the communication domain, obtained through the `get_hccl_comm_name` API.
- **`ep_world_size`** (`int`): Required. Total number of ranks in the communication domain. The value can be `2`, `4`, `8`, `16`, `32`, `64`, `128`, or `256`.
- **`send_counts`** (`List[int]`): Required. Integer list with length `e * ep_world_size`, representing the number of tokens sent from the local device to each destination device. Assuming that the destination device ID is `i` ($0 \le i < ep_world_size$) and the expert ID is `j` ($0 \le j < e$), `send_counts[i][j]` represents the number of tokens sent from the local device to the `j`-th expert on device `i`. The length must be `e * ep_world_size`, and all elements must be non-negative integers.
- **`recv_counts`** (`List[int]`): Required. Integer list with length `e * ep_world_size`, representing the number of tokens received by the local device from each source device. Assuming that the source device ID is `i` ($0 \le i < ep_world_size$) and the expert ID is `j` ($0 \le j < e$), `recv_counts[i][j]` represents the number of tokens received by the local device from the `j`-th expert on device `i`. The length must be `e * ep_world_size`, and all elements must be non-negative integers.
- **`gmm_y_dtype`** (`int`): Required. Data type of output tensor `gmm_y` of the routed expert `GroupedMatMul` computation. The data type can be `float16` or `bfloat16`.
- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`send_counts_tensor`** (`Tensor`): Optional. Currently, only `None` is supported.
- **`recv_counts_tensor`** (`Tensor`): Optional. Currently, only `None` is supported.
- **`mm_x`** (`Tensor`): Optional. Left matrix input of shared expert `MatMul`, provided only when shared experts are enabled. The data type can be `hifloat8`, `float8_e4m3fn`, `float8_e5m2`, or `float4_e2m1fn_x2`, and must be the same as the data type of `gmm_x`. This parameter must be a 2D tensor with shape $(BS, H2)$. The data layout can be `ND`. When the data type is `float4_e2m1fn_x2`, inner dimension `H2` must be even and cannot be `2` to ensure that 8 bits can be converted into two `float4_e2m1fn_x2` elements. The default value is `None`.
- **`mm_weight`** (`Tensor`): Optional. Right matrix input of shared expert `MatMul`, provided only when shared experts are enabled. The data type can be `hifloat8`, `float8_e4m3fn`, `float8_e5m2`, or `float4_e2m1fn_x2`, and must be the same as the data type of `gmm_weight`. This parameter must be a 2D tensor with shape $(H2, N2)$. The data layout can be `ND`. In the MX quantization scenario, when both `mm_x` and `mm_weight` use the `float4_e2m1fn_x2` data type, only inference scenarios are supported. In this case, inner dimension `H2` of `mm_x` must be even and cannot be `2`. When `mm_weight` is not transposed, inner dimension `N2` must be even; when it is transposed, inner dimension `H2` must be even and cannot be `2`, to ensure that 8 bits can be converted into two `float4_e2m1fn_x2` elements. The default value is `None`.
- **`mm_x_scale`** (`Tensor`): Optional. Quantization scale of shared expert left matrix `mm_x`. The data type can be `float32` or `float8_e8m0fnu`. In the per-tensor quantization scenario, this parameter must be a 1D tensor with shape $(1)$. In the MX quantization scenario, this parameter must be a 3D tensor with shape $(BS, \operatorname{ceil}(H2/64), 2)$. The data layout can be `ND`. The default value is `None`.
- **`mm_weight_scale`** (`Tensor`): Optional. Quantization scale of shared expert right matrix `mm_weight`. The data type can be `float32` or `float8_e8m0fnu`. In the per-tensor quantization scenario, this parameter must be a 1D tensor with shape $(1)$. In the MX quantization scenario, this parameter must be a 3D tensor with shape $(\operatorname{ceil}(H2/64), N2, 2)$. The data layout can be `ND`. The default value is `None`.
- **`gmm_x_quant_mode`** (`int`): Optional. Quantization mode of routed expert left matrix. In the current version, the value can be `1` or `6`, representing per-tensor quantization and MX quantization, respectively.
- **`gmm_weight_quant_mode`** (`int`): Optional. Quantization mode of routed expert right matrix. In the current version, the value can be `1` or `6`, representing per-tensor quantization and MX quantization, respectively.
- **`mm_x_quant_mode`** (`int`): Optional. Quantization mode of shared expert left matrix. In the current version, the value can be `1` or `6`, representing per-tensor quantization and MX quantization, respectively.
- **`mm_weight_quant_mode`** (`int`): Optional. Quantization mode of shared expert right matrix. In the current version, the value can be `1` or `6`, representing per-tensor quantization and MX quantization, respectively.
- **`permute_out_flag`** (`bool`): Optional. Specifies whether to return the rearranged routed expert matrix after communication, that is, `permute_out`. The default value is `False`. When set to `True`, the return value includes this tensor.
- **`group_size`** (`List[int]`): Optional. Number of elements in the corresponding dimension of the `gmm_x`, `gmm_weight`, `mm_x`, and `mm_weight` inputs that can be quantized using one value in the corresponding `gmm_x_scale`, `gmm_weight_scale`, `mm_x_scale`, or `mm_weight_scale` input. The shape of `group_size` is `[groupSizeM, groupSizeN, groupSizeK]`. `groupSizeM`, `groupSizeN`, and `groupSizeK` represent the number of elements in the corresponding dimensions that can be quantized using one quantization scale.

  The default value is `[0, 0, 0]`. The value of `group_size` is effective only in the MX quantization scenario. In other scenarios, `[0, 0, 0]` must be passed. For configuration principles, see the constraints.

- **`gmm_x_dtype`** (`int`): Optional. Actual data type of routed expert left matrix `gmm_x`. For data types not natively supported by PyTorch, such as `torch_npu.hifloat8` and `torch_npu.float4_e2m1fn_x2`, this parameter must be specified. The default value is `None`.
- **`gmm_weight_dtype`** (`int`): Optional. Actual data type of routed expert right matrix `gmm_weight`. For data types not natively supported by PyTorch, such as `torch_npu.hifloat8` and `torch_npu.float4_e2m1fn_x2`, this parameter must be specified. The default value is `None`.
- **`gmm_x_scale_dtype`** (`int`): Optional. Actual data type of routed expert quantization scale `gmm_x_scale`. For data types not natively supported by PyTorch, such as `torch_npu.float8_e8m0fnu`, this parameter must be specified. The default value is `None`.
- **`gmm_weight_scale_dtype`** (`int`): Optional. Actual data type of routed expert quantization scale `gmm_weight_scale`. For data types not natively supported by PyTorch, such as `torch_npu.float8_e8m0fnu`, this parameter must be specified. The default value is `None`.
- **`mm_x_dtype`** (`int`): Optional. Data type of shared expert left matrix `mm_x`. When shared expert computation is performed, this parameter must be specified for data types not natively supported by PyTorch, such as `torch_npu.hifloat8` and `torch_npu.float4_e2m1fn_x2`.
- **`mm_weight_dtype`** (`int`): Optional. Data type of shared expert right matrix `mm_weight`. When shared expert computation is performed, this parameter must be specified for data types not natively supported by PyTorch, such as `torch_npu.hifloat8` and `torch_npu.float4_e2m1fn_x2`.
- **`mm_x_scale_dtype`** (`int`): Optional. Data type of shared expert quantization scale `mm_x_scale`. When shared expert computation is performed, this parameter must be specified for data types not natively supported by PyTorch, such as `torch_npu.float8_e8m0fnu`.
- **`mm_weight_scale_dtype`** (`int`): Optional. Data type of shared expert quantization scale `mm_weight_scale`. When shared expert computation is performed, this parameter must be specified for data types not natively supported by PyTorch, such as `torch_npu.float8_e8m0fnu`.
- **`mm_y_dtype`** (`int`): Optional. Data type of shared expert output tensor `mm_y`. When shared expert computation is performed, this parameter must be specified. The data type can be `float16` or `bfloat16`.
- **`comm_mode`** (`str`): Optional. Communication mode. The value can be `"ai_cpu"`, `"ccu"`, or `None`. When set to `None`, AI_CPU communication is used by default. The default value is `None`.

## Return Values

- **`gmm_y`** (`Tensor`): Output of routed expert `GroupedMatMul` computation. The data type is specified by `gmm_y_dtype` and can be `float16` or `bfloat16`. This parameter must be a 2D tensor with shape `(A, N1)`. The data layout can be `ND`.
- **`mm_y`** (`Tensor`): Output of shared expert `MatMul`. The data type is specified by `mm_y_dtype` and can be `float16` or `bfloat16`, and must be the same as the data type of `gmm_y`. This parameter must be a 2D tensor with shape `(BS, N2)`. This output is returned only when `mm_x` and `mm_weight` are provided. The data layout can be `ND`.
- **`permute_out`** (`Tensor`): Computation output after `Permute`. The data type is the same as that of `gmm_x`. This parameter must be a 2D tensor with shape `(A, H1)`. The data layout can be `ND`.

## Constraints

- This API can be used in training and inference scenarios.
- **Communication engine constraints**: CCU and AI_CPU communication are supported and can be configured through the `comm_mode` parameter. When `comm_mode` is `None`, AI_CPU communication is used.
- This API supports single-operator calls and graph-mode calls in the T-T quantization scenario.
- The variables used in the Shape descriptions are defined as follows:
    - `BS` represents the batch sequence size.
    - `K` represents the number of selected top-K experts. When shared expert computation is enabled, `K` must be in the range `[2, 8]`.
    - `BSK = sum(send_counts)` represents the total number of tokens sent from the local device to other devices during `AlltoAllv` communication. The value range is `(0, 52428800)`.
    - `H1` represents the hidden size of routed experts on the local device. The value range is `(0, 65536)`.
    - `H2` represents the hidden size of shared experts on the local device. The value range is `(0, 12288]`.
    - `N1` represents the output dimension of routed experts. The value range is `(0, 65536)`.
    - `N2` represents the output dimension of shared experts. The value range is `(0, 65536)`.
    - `e` represents the number of experts on each device after communication. The value range is `(0, 32]`, and `e * ep_world_size` must be less than or equal to `256`.
    - `A` represents the total number of tokens in the output of routed expert computation. `A = sum(recv_counts)`. The sum of `A` across all devices in the EP communication domain is equal to the sum of `BSK` across all devices.
    - For data sent from device `i` to device `j`, the amount specified by `send_counts[j]` must equal the amount specified by `recv_counts[i]` for data received by device `j`.
- The relationships between the values of `gmm_x_quant_mode`, `gmm_weight_quant_mode`, `mm_x_quant_mode`, and `mm_weight_quant_mode` and the quantization modes are as follows:
    - `0`: Non-quantization
    - `1`: `pertensor` quantization
    - `2`: `perchannel` quantization
    - `3`: `pertoken` quantization
    - `4`: `pergroup` quantization
    - `5`: `perblock` quantization
    - `6`: MX quantization
    - `7`: Dynamic `pertoken` quantization
    - Currently, only `[1, 1]` and `[6, 6]` are supported as combinations of `gmm_x_quant_mode` and `gmm_weight_quant_mode`, representing T-T quantization and MX quantization, respectively.
    - Currently, only `[1, 1]` and `[6, 6]` are supported as combinations of `mm_x_quant_mode` and `mm_weight_quant_mode`, representing T-T quantization and MX quantization, respectively. The quantization combination must be consistent with the combination of `gmm_x_quant_mode` and `gmm_weight_quant_mode`.
- **`group_sizes`**:
    - When one or more of `groupSizeM`, `groupSizeN`, and `groupSizeK` are `0`, their values are reset based on the shapes of input `gmm_x_scale`, `gmm_weight_scale`, `mm_x_scale`, `mm_weight_scale`, `gmm_x`, `gmm_weight`, `mm_x`, and `mm_weight` for computation.
    - Configuration principles are as follows: when `groupSizeM = 0`, the quantization grouping value along the `m` dimension is inferred by the API using `groupSizeM = m / scaleM`, where `m` must be divisible by `scaleM`. Here, `m` corresponds to the `m` dimension of `gmm_x` and `mm_x`, and `scaleM` corresponds to the `m` dimension of `gmm_x_scale` and `mm_x_scale`. When `groupSizeK = 0`, the quantization grouping value along the `k` dimension is inferred by the API using `groupSizeK = k / scaleK`, where `k` must be divisible by `scaleK`. Here, `k` corresponds to the `k` dimension of `gmm_x` and `mm_x`, and `scaleK` corresponds to the `k` dimension of `gmm_x_scale` and `mm_x_scale`. When `groupSizeN = 0`, the quantization grouping value along the `n` dimension is inferred by the API using `groupSizeN = n / scaleN`, where `n` must be divisible by `scaleN`. Here, `n` corresponds to the `n` dimension of `gmm_weight` and `mm_weight`, and `scaleN` corresponds to the `n` dimension of `gmm_weight_scale` and `mm_weight_scale`.
    - When the conditions for resetting the values are met, if `gmm_x_scale`, `mm_x_scale`, and `mm_weight_scale` are all 3D tensors, `gmm_weight_scale` is a 4D tensor, and all data types are `float8_e8m0fnu`, the inferred combination of `[groupSizeM, groupSizeN, groupSizeK]` is `[1, 1, 32]`.
- When all output tensors (`gmm_y`, `mm_y`, and `permute_out`) on a device are empty tensors, `torch.distributed.barrier()` must be called explicitly to ensure that the process on that device synchronizes and waits for other devices to complete communication and computation. If synchronization is not added, `AlltoAllv` communication will be blocked due to process desynchronization.

- Detailed input and output data type constraints for each quantization mode are provided in the following tables.

    **Table 1** Data type constraints for T-T quantization

    | gmm_x | gmm_weight | gmm_x_scale | gmm_weight_scale | gmm_x_quant_mode/gmm_weight_quant_mode | gmm_y | mm_x | mm_weight | mm_x_scale | mm_weight_scale | mm_x_quant_mode/mm_weight_quant_mode | mm_y |
    |---------|--------|--------|--------|--------|--------|---------|--------|--------|--------|--------|--------|
    | hifloat8 | hifloat8 | float32 | float32 | [1, 1] | float16 | hifloat8 | hifloat8 | float32 | float32 | [1, 1] | float16 |
    | hifloat8 | hifloat8 | float32 | float32 | [1, 1] | bfloat16 | hifloat8 | hifloat8 | float32 | float32 | [1, 1] | bfloat16 |

    **Table 2** Data type constraints for MX quantization

    | gmm_x | gmm_weight | gmm_x_scale | gmm_weight_scale | gmm_x_quant_mode/gmm_weight_quant_mode | gmm_y | mm_x | mm_weight | mm_x_scale | mm_weight_scale | mm_x_quant_mode/mm_weight_quant_mode | mm_y |
    |---------|--------|--------|--------|--------|--------|---------|--------|--------|--------|--------|--------|
    | float8_e4m3fn | float8_e4m3fn | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | float16 | float8_e4m3fn | float8_e4m3fn | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | float16 |
    | float8_e4m3fn | float8_e4m3fn | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | bfloat16 | float8_e4m3fn | float8_e4m3fn | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | bfloat16 |
    | float8_e4m3fn | float8_e5m2 | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | float16 | float8_e4m3fn | float8_e5m2 | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | float16 |
    | float8_e4m3fn | float8_e5m2 | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | bfloat16 | float8_e4m3fn | float8_e5m2 | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | bfloat16 |
    | float8_e5m2 | float8_e5m2 | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | float16 | float8_e5m2 | float8_e5m2 | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | float16 |
    | float8_e5m2 | float8_e5m2 | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | bfloat16 | float8_e5m2 | float8_e5m2 | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | bfloat16 |
    | float8_e5m2 | float8_e4m3fn | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | float16 | float8_e5m2 | float8_e4m3fn | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | float16 |
    | float8_e5m2 | float8_e4m3fn | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | bfloat16 | float8_e5m2 | float8_e4m3fn | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | bfloat16 |
    | float4_e2m1fn_x2 | float4_e2m1fn_x2 | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | float16 | float4_e2m1fn_x2 | float4_e2m1fn_x2 | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | float16 |
    | float4_e2m1fn_x2 | float4_e2m1fn_x2 | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | bfloat16 | float4_e2m1fn_x2 | float4_e2m1fn_x2 | float8_e8m0fnu | float8_e8m0fnu | [6, 6] | bfloat16 |

## Examples

- Single-operator call

    - T-T quantization scenario

        ```python
        import torch
        import torch_npu
        import torch.distributed as dist
        import torch.multiprocessing as mp
        import numpy as np

        def generate_counts(ep_world_size, e, total_tokens, seed=None):
            np.random.seed(seed if seed is not None else 42)
            per_rank_total = total_tokens
            base = per_rank_total // (ep_world_size * e)
            remainder = per_rank_total % (ep_world_size * e)
            send_counts = [base] * (ep_world_size * e)
            for i in range(remainder):
                send_counts[-1 - i] += 1
            recv_counts = send_counts.copy()
            return send_counts, recv_counts

        def run_npu_alltoallv_quant_gmm(rank, world_size, master_ip, master_port):
            torch_npu.npu.set_device(rank)
            init_method = f"tcp://{master_ip}:{master_port}"
            dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
            from torch.distributed.distributed_c10d import _get_default_group
            default_pg = _get_default_group()
            if torch.__version__ > "2.0.1":
                hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
            else:
                hcom_info = default_pg.get_hccl_comm_name(rank)

            BS = 128
            K = 2
            e = 2
            H1, N1 = 256, 256
            H2, N2 = 256, 128
            total_tokens = BS * K
            send_counts, recv_counts = generate_counts(world_size, e, total_tokens, seed=rank)
            gmm_x = torch.randint(0, 255, (total_tokens, H1), dtype=torch.uint8).npu()
            gmm_weight = torch.randint(0, 255, (e, H1, N1), dtype=torch.uint8).npu()
            gmm_x_scale = torch.tensor([0.5], dtype=None).npu()
            gmm_weight_scale = torch.tensor([0.3], dtype=None).npu()
            mm_x = torch.randint(0, 255, (BS, H2), dtype=torch.uint8).npu()
            mm_weight = torch.randint(0, 255, (H2, N2), dtype=torch.uint8).npu()
            mm_x_scale = torch.tensor([0.4], dtype=None).npu()
            mm_weight_scale = torch.tensor([0.2], dtype=None).npu()
            quant_mode = 1
            out_dtype = torch.float16

            gmm_y, mm_y, permute_out = torch_npu.npu_alltoallv_quant_gmm(
                gmm_x=gmm_x,
                gmm_weight=gmm_weight,
                gmm_x_scale=gmm_x_scale,
                gmm_weight_scale=gmm_weight_scale,
                hcom=hcom_info,
                ep_world_size=world_size,
                send_counts=send_counts,
                recv_counts=recv_counts,
                gmm_y_dtype=out_dtype,
                mm_x=mm_x,
                mm_weight=mm_weight,
                mm_x_scale=mm_x_scale,
                mm_weight_scale=mm_weight_scale,
                gmm_x_quant_mode=quant_mode,
                gmm_weight_quant_mode=quant_mode,
                mm_x_quant_mode=quant_mode,
                mm_weight_quant_mode=quant_mode,
                permute_out_flag=True,
                gmm_x_dtype=torch_npu.hifloat8,
                gmm_weight_dtype=torch_npu.hifloat8,
                gmm_x_scale_dtype=None,
                gmm_weight_scale_dtype=None,
                mm_x_dtype=torch_npu.hifloat8,
                mm_weight_dtype=torch_npu.hifloat8,
                mm_x_scale_dtype=None,
                mm_weight_scale_dtype=None,
                mm_y_dtype=out_dtype,
                send_counts_tensor=None,
                recv_counts_tensor=None,
                group_size=None
            )

        if __name__ == "__main__":
            world_size = 2
            master_ip = "127.0.0.1"
            master_port = "50001"
            mp.spawn(run_npu_alltoallv_quant_gmm, args=(world_size, master_ip, master_port), nprocs=world_size, join=True)
        ```

    - mx quantization scenario (mxfp8)

        ```python
        import torch
        import torch_npu
        import torch.distributed as dist
        import torch.multiprocessing as mp
        import numpy as np
        import math

        def generate_counts(ep_world_size, e, total_tokens, seed=None):
            np.random.seed(seed if seed is not None else 42)
            per_rank_total = total_tokens
            base = per_rank_total // (ep_world_size * e)
            remainder = per_rank_total % (ep_world_size * e)
            send_counts = [base] * (ep_world_size * e)
            for i in range(remainder):
                send_counts[-1 - i] += 1
            recv_counts = send_counts.copy()
            return send_counts, recv_counts

        def run_npu_alltoallv_quant_gmm(rank, world_size, master_ip, master_port):
            torch_npu.npu.set_device(rank)
            init_method = f"tcp://{master_ip}:{master_port}"
            dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
            from torch.distributed.distributed_c10d import _get_default_group
            default_pg = _get_default_group()
            if torch.__version__ > "2.0.1":
                hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
            else:
                hcom_info = default_pg.get_hccl_comm_name(rank)

            BS = 128
            K = 2
            e = 2
            H1, N1 = 256, 256
            H2, N2 = 256, 128
            total_tokens = BS * K
            send_counts, recv_counts = generate_counts(world_size, e, total_tokens, seed=rank)
            gmm_x = torch.ones(total_tokens, H1, dtype=torch.int8).to(torch.float8_e4m3fn).npu()
            gmm_weight = torch.ones(e, H1, N1, dtype=torch.int8).to(torch.float8_e5m2).npu()
            gmm_x_scale = torch.ones(total_tokens, math.ceil(H1 / 64), 2, dtype=torch.int8).npu()
            gmm_weight_scale = torch.ones(e, math.ceil(H1 / 64), N1, 2, dtype=torch.int8).npu()
            mm_x = torch.ones(BS, H2, dtype=torch.int8).to(torch.float8_e4m3fn).npu()
            mm_weight = torch.ones(H2, N2, dtype=torch.int8).to(torch.float8_e5m2).npu()
            mm_x_scale = torch.ones(BS, math.ceil(H2 / 64), 2, dtype=torch.int8).npu()
            mm_weight_scale = torch.ones(math.ceil(H2 / 64), N2, 2, dtype=torch.int8).npu()
            quant_mode = 6
            out_dtype = torch.float16

            gmm_y, mm_y, permute_out = torch_npu.npu_alltoallv_quant_gmm(
                gmm_x=gmm_x,
                gmm_weight=gmm_weight,
                gmm_x_scale=gmm_x_scale,
                gmm_weight_scale=gmm_weight_scale,
                hcom=hcom_info,
                ep_world_size=world_size,
                send_counts=send_counts,
                recv_counts=recv_counts,
                gmm_y_dtype=out_dtype,
                mm_x=mm_x,
                mm_weight=mm_weight,
                mm_x_scale=mm_x_scale,
                mm_weight_scale=mm_weight_scale,
                gmm_x_quant_mode=quant_mode,
                gmm_weight_quant_mode=quant_mode,
                mm_x_quant_mode=quant_mode,
                mm_weight_quant_mode=quant_mode,
                permute_out_flag=True,
                gmm_x_dtype=None,
                gmm_weight_dtype=None,
                gmm_x_scale_dtype=torch_npu.float8_e8m0fnu,
                gmm_weight_scale_dtype=torch_npu.float8_e8m0fnu,
                mm_x_dtype=None,
                mm_weight_dtype=None,
                mm_x_scale_dtype=torch_npu.float8_e8m0fnu,
                mm_weight_scale_dtype=torch_npu.float8_e8m0fnu,
                mm_y_dtype=out_dtype,
                send_counts_tensor=None,
                recv_counts_tensor=None,
                group_size=None
            )

        if __name__ == "__main__":
            world_size = 2
            master_ip = "127.0.0.1"
            master_port = "50001"
            mp.spawn(run_npu_alltoallv_quant_gmm, args=(world_size, master_ip, master_port), nprocs=world_size, join=True)
        ```

    - mx quantization scenario (mxfp4)

        ```python
        import torch
        import torch_npu
        import torch.distributed as dist
        import torch.multiprocessing as mp
        import numpy as np
        import math

        def generate_counts(ep_world_size, e, total_tokens, seed=None):
            np.random.seed(seed if seed is not None else 42)
            per_rank_total = total_tokens
            base = per_rank_total // (ep_world_size * e)
            remainder = per_rank_total % (ep_world_size * e)
            send_counts = [base] * (ep_world_size * e)
            for i in range(remainder):
                send_counts[-1 - i] += 1
            recv_counts = send_counts.copy()
            return send_counts, recv_counts

        def run_npu_alltoallv_quant_gmm(rank, world_size, master_ip, master_port):
            torch_npu.npu.set_device(rank)
            init_method = f"tcp://{master_ip}:{master_port}"
            dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
            from torch.distributed.distributed_c10d import _get_default_group
            default_pg = _get_default_group()
            if torch.__version__ > "2.0.1":
                hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
            else:
                hcom_info = default_pg.get_hccl_comm_name(rank)

            BS = 128
            K = 2
            e = 2
            H1, N1 = 256, 256
            H2, N2 = 256, 128
            total_tokens = BS * K
            send_counts, recv_counts = generate_counts(world_size, e, total_tokens, seed=rank)
            # Use `uint8` to represent two MXFP4 elements, where the inner dimension of `uint8` must be doubled to obtain the actual inner dimension of MXFP4.
            gmm_x = torch.ones(total_tokens, int(H1/2), dtype=torch.uint8).to(torch.float8_e4m3fn).npu()
            gmm_weight = torch.ones(e, H1, int(N1/2), dtype=torch.uint8).to(torch.float8_e5m2).npu()
            gmm_x_scale = torch.ones(total_tokens, math.ceil(H1 / 64), 2, dtype=torch.int8).npu()
            gmm_weight_scale = torch.ones(e, math.ceil(H1 / 64), N1, 2, dtype=torch.int8).npu()
            mm_x = torch.ones(BS, int(H2/2), dtype=torch.uint8).to(torch.float8_e4m3fn).npu()
            mm_weight = torch.ones(H2, int(N2/2), dtype=torch.uint8).to(torch.float8_e5m2).npu()
            mm_x_scale = torch.ones(BS, math.ceil(H2 / 64), 2, dtype=torch.uint8).npu()
            mm_weight_scale = torch.ones(math.ceil(H2 / 64), N2, 2, dtype=torch.int8).npu()
            quant_mode = 6
            out_dtype = torch.float16

            gmm_y, mm_y, permute_out = torch_npu.npu_alltoallv_quant_gmm(
                gmm_x=gmm_x,
                gmm_weight=gmm_weight,
                gmm_x_scale=gmm_x_scale,
                gmm_weight_scale=gmm_weight_scale,
                hcom=hcom_info,
                ep_world_size=world_size,
                send_counts=send_counts,
                recv_counts=recv_counts,
                gmm_y_dtype=out_dtype,
                mm_x=mm_x,
                mm_weight=mm_weight,
                mm_x_scale=mm_x_scale,
                mm_weight_scale=mm_weight_scale,
                gmm_x_quant_mode=quant_mode,
                gmm_weight_quant_mode=quant_mode,
                mm_x_quant_mode=quant_mode,
                mm_weight_quant_mode=quant_mode,
                permute_out_flag=True,
                gmm_x_dtype=torch_npu.float4_e2m1fn_x2,
                gmm_weight_dtype=torch_npu.float4_e2m1fn_x2,
                gmm_x_scale_dtype=torch_npu.float8_e8m0fnu,
                gmm_weight_scale_dtype=torch_npu.float8_e8m0fnu,
                mm_x_dtype=torch_npu.float4_e2m1fn_x2,
                mm_weight_dtype=torch_npu.float4_e2m1fn_x2,
                mm_x_scale_dtype=torch_npu.float8_e8m0fnu,
                mm_weight_scale_dtype=torch_npu.float8_e8m0fnu,
                mm_y_dtype=out_dtype,
                send_counts_tensor=None,
                recv_counts_tensor=None,
                group_size=None
            )

        if __name__ == "__main__":
            world_size = 2
            master_ip = "127.0.0.1"
            master_port = "50001"
            mp.spawn(run_npu_alltoallv_quant_gmm, args=(world_size, master_ip, master_port), nprocs=world_size, join=True)
        ```

- Graph mode call

    - T-T quantization scenario

        ```python
        import torch
        import torch_npu
        import torch.distributed as dist
        import torch.multiprocessing as mp
        import torchair
        import numpy as np
        from en_dtypes import hifloat8

        class ALLTOALLV_GMM_GRAPH_Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, _gmm_x, _gmm_weight, _gmm_x_scale, _gmm_weight_scale, _hcom, _ep_world_size,
                        _send_counts, _recv_counts, _gmm_y_dtype, _mm_y_dtype, _mm_x, _mm_weight, _mm_x_scale,
                        _mm_weight_scale, _permute_out_flag, _gmm_x_quant_mode, _gmm_weight_quant_mode,
                        _mm_x_quant_mode, _mm_weight_quant_mode, _gmm_x_dtype, _gmm_weight_dtype, _mm_x_dtype,
                        _mm_weight_dtype, _gmm_weight_scale_dtype=None, _gmm_x_scale_dtype=None,
                        _mm_x_scale_dtype=None, _mm_weight_scale_dtype=None, _trans_gmm_weight=False,
                        _trans_mm_weight=False):
                if _trans_gmm_weight:
                    _gmm_weight = torch.transpose(_gmm_weight, -2, -1)
                if _trans_mm_weight and _mm_weight is not None:
                    _mm_weight = _mm_weight.t()
                gmm_y, mm_y, permute_out = torch_npu.npu_alltoallv_quant_gmm(
                    gmm_x=_gmm_x,
                    gmm_weight=_gmm_weight,
                    gmm_x_scale=_gmm_x_scale,
                    gmm_weight_scale=_gmm_weight_scale,
                    hcom=_hcom,
                    ep_world_size=_ep_world_size,
                    send_counts=_send_counts,
                    recv_counts=_recv_counts,
                    gmm_y_dtype=_gmm_y_dtype,
                    mm_y_dtype=_mm_y_dtype,
                    mm_x=_mm_x,
                    mm_weight=_mm_weight,
                    mm_x_scale=_mm_x_scale,
                    mm_weight_scale=_mm_weight_scale,
                    permute_out_flag=_permute_out_flag,
                    gmm_x_quant_mode=_gmm_x_quant_mode,
                    gmm_weight_quant_mode=_gmm_weight_quant_mode,
                    mm_x_quant_mode=_mm_x_quant_mode,
                    mm_weight_quant_mode=_mm_weight_quant_mode,
                    gmm_x_dtype=_gmm_x_dtype,
                    gmm_weight_dtype=_gmm_weight_dtype,
                    mm_x_dtype=_mm_x_dtype,
                    mm_weight_dtype=_mm_weight_dtype
                )
                return gmm_y, mm_y, permute_out

        def run_npu_alltoallv_gmm(rank, ep_world_size, master_ip, master_port, gmm_x, gmm_w, send_counts, recv_counts, dtype,
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

            model = ALLTOALLV_GMM_GRAPH_Model()
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            # Static graph: dynamic=False dynamic graph: dynamic=True
            model = torch.compile(model, backend=npu_backend, dynamic=False)
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
                        _permute_out_flag=False,
                        _gmm_x_quant_mode=1,
                        _gmm_weight_quant_mode=1,
                        _mm_x_quant_mode=None,
                        _mm_weight_quant_mode=None,
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
            epWorldSize = 2
            e = 4
            master_ip = '127.0.0.1'
            master_port = '50001'
            BS = 512
            K = 8
            gmm_x_shape = [BS*K, 2048]
            gmm_weight_shape = [e, 2048, 2048]
            send_counts = [512] * (e * epWorldSize)
            recv_counts = [512] * (e * epWorldSize)
            scale_shape = [1]
            dtype = torch_npu.hifloat8
            mp.spawn(run_npu_alltoallv_gmm, args=(epWorldSize, master_ip, master_port, gmm_x_shape, gmm_weight_shape, send_counts, recv_counts, dtype, scale_shape, scale_shape), nprocs=epWorldSize)
        ```
