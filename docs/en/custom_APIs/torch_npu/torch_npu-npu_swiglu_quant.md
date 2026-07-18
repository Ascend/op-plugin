# torch_npu.npu_swiglu_quant

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |

## Function

- Description: Adds a quantization operation after the SwiGLU activation function to perform `SwiGluQuant` computation on the input `x`. This API supports `int8` or `int4` quantized outputs, MoE and non-MoE scenarios (when `group_index` is omitted), group quantization, and dynamic or static quantization.
- Formulas:
    - SwiGLU activation: Perform SwiGLU on `x`. `activate_left` controls left or right activation. The following example shows the left activation equation:
        $$
        swiglu(x)=swish(x[:,0:N])*x[:,N:2N]
        $$

    - Grouping (`cumsum` or `count` modes are supported):

        Perform group-wise computation on the result of applying SwiGLU to the input `x`. `group_index` indicates the number of tokens in each group. Each group uses different `smooth_scales` for quantization.

        For example, assume that `x` has shape `[6, 2N]`. In `cumsum` mode, `group_index` has shape `[2, 4, 6]`, which indicates that there are three groups. The corresponding `smooth_scales` has shape `[3, H]`. The data in each group is quantized using different `smooth_scales`.

        - group0=x\[0:2, :\], scale0=smooth_scales\[0, :\]
        - group1=x\[2:4, :\], scale1=smooth_scales\[1, :\]
        - group2=x\[4:6, :\], scale2=smooth_scales\[2, :\]

        In `count` mode, `group_index` must be set to `[2, 2, 2]`, which indicates that each group contains two tokens.

    - Quantization:
        1. Perform smooth quantization, where `out` is the output of the SwiGLU activation in the previous step.
            $$
            out=out*smooth\_scales
            $$

        2. Perform dynamic or static quantization on the activation result. The following example shows the dynamic quantization (`dynamic_quant`) equation. For detailed mathematical formulas, see [aclnnSwiGluQuantV2](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/aolapi/context/ops-nn/aclnnSwiGluQuantV2.md).
            $$
            out,scale=dynamic\_quant(out)
            $$

## Prototype

```python
torch_npu.npu_swiglu_quant(x, *, smooth_scales=None, offsets=None, group_index=None, activate_left=False, quant_mode=0, group_list_type=0, dst_type=None) -> (Tensor, Tensor)
```

## Parameters

> [!NOTE]  
> Variables used in tensor shapes:
>
> - `G`: The number of `group_index` groups. The value is greater than 0.
> - `N`: Half the size of the last dimension of the input `x`. The value is greater than 0.

- **`x`** (`Tensor`): Required. Target input tensor. The data type can be `float16`, `bfloat16`, or `float32`. Non-contiguous tensors are supported. The data layout is ND. The number of dimensions must be greater than 1. The size of the last dimension must be even and cannot exceed 8192. When `dst_type` is `int4`, the last dimension of `x` must be a multiple of 4.
- **`*`**: Required. Positional parameter separator. Parameters before it are positional parameters and must be provided in order. Parameters after it are optional keyword parameters and must be specified using key-value pairs. If omitted, default values are used.
- **`smooth_scales`** (`Tensor`): Optional. Smooth quantization coefficients. The data type can be `float32`. Non-contiguous tensors are supported. The data layout is ND. The shape can be `[G, N]` or `[G,]`.
- **`offsets`** (`Tensor`): Optional. Quantization offsets. This parameter does not take effect in dynamic quantization scenarios. Pass `None`. In static quantization scenarios: The data type can be `float`. Non-contiguous tensors are supported. The data layout is ND. The shape must be `[G, N]` in `per_channel` mode and `[G]` in `per_tensor` mode. The data type and shape must be identical to those of `smooth_scales`.
- **`group_index`** (`Tensor`): Optional. Currently, `cumsum` and `count` modes are supported. The data type can be `int32`. The data layout is ND. This parameter must be 1D with shape `[G]`. Elements within `group_index` must be non-decreasing, and the maximum value cannot exceed the product of all dimensions of input `x` except the last dimension.
- **`activate_left`** (`bool`): Optional. Specifies whether to perform left activation during the SwiGLU process. The default value is `False`.
- **`quant_mode`** (`int`): Optional. The quantization type. The default value is `0`. Valid values are `0` (static quantization) or `1` (dynamic quantization).
- **`group_list_type`** (`int`): Optional. Type of `group_index`. The default value is `0`. Valid values are `0` (`cumsum` mode) or `1` (`count` mode).
- **`dst_type`** (`ScalarType`): Optional. Output quantization data type. Valid values are `int8` or `int4`. If `None` is passed, it is treated as `int8`. The default value is `None`.

## Return Values

- **`out`** (`Tensor`): Quantized output tensor. The data type can be `int8` or `int4`. Non-contiguous tensors are supported. The data layout is ND.
- **`scale`** (`Tensor`): Quantization scale. Compared with the input `x`, the output `scale` lacks the final dimension, while all other dimensions remain identical to those of `x`. The data type can be `float32`. The data layout is ND.

## Example

```python
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestNPUSwigluQuant(TestCase):
    def test_npu_swiglu_quant(self, device="npu"):
        batch_size, hidden_size = 4608, 2048
        num_groups = 8
        group_size = batch_size // num_groups
        
        npu_x = torch.randn(batch_size, hidden_size, dtype=torch.float32, device=device)
        g_index = torch.tensor([(i + 1) * group_size for i in range(num_groups)], dtype=torch.int32, device=device)
        s_scales = torch.randn(num_groups, hidden_size // 2, dtype=torch.float32, device=device)
        offsets = torch.randn(num_groups, hidden_size // 2, dtype=torch.float32, device=device)

        result = torch_npu.npu_swiglu_quant(
            npu_x,
            smooth_scales=s_scales,
            offsets=offsets,
            group_index=g_index,
            activate_left=False,
            quant_mode=1,
            group_list_type=0,
            dst_type=torch.int8
        )


if __name__ == "__main__":
    run_tests()
```
