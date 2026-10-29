# torch_npu.npu_fused_matmul

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[Interface Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#interface-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- **API function**: Fuses matrix multiplication with general vector computation to reduce data movement and improve performance.

- **Formulas**:

  $$
  y = \operatorname{op}((x1 @ x2 + bias),\ x3)
  $$

  The operation type is defined by the `fused_op_type` input and supports the following operations.

  | fused_op_type | Formula |
  | :--- | :--- |
  | add | $y = (x1 @ x2 + bias) + x3$ |
  | mul | $y = (x1 @ x2 + bias) \times x3$ |
  | gelu_erf | $y = \operatorname{gelu\_erf}(x1 @ x2)$ |
  | gelu_tanh | $y = \operatorname{gelu\_tanh}(x1 @ x2)$ |
  | relu | $y = \operatorname{relu}(x1 @ x2 + bias)$ |

  - `x1` is the left matrix of the matrix multiplication.
  - `x2` is the right matrix of the matrix multiplication.
  - `x3` is the matrix for the vector computation and takes effect when `fused_op_type` is `"add"` or `"mul"`.
  - `bias` is the bias of the matrix multiplication.
  - `fused_op_type` is an input string and supports `""` (no fusion), `"16cast32"` (no fusion, and the output is `torch.float32`; when the input is `torch.float16` or `torch.bfloat16`, the output is converted to `torch.float32`), `"mul"`, `"gelu_erf"`, `"gelu_tanh"`, and `"relu"`.

## Prototype

```python
torch_npu.npu_fused_matmul(x1, x2, *, bias=None, x3=None, fused_op_type) -> Tensor
```

## Parameters

- **`x1`** (`Tensor`): Required. `x1` in the matrix multiplication. The data layout can be ND. Non-contiguous Tensors are supported. When `fused_op_type` is `"relu"` or `""`, the input can have 2 to 6 dimensions. When it is `"add"` or `"mul"`, the input can be 2D (M,K) or 3D (B,M,K). In other cases, the input has 2 dimensions (M, K). In multi-dimensional scenarios, the batch dimension must be the same when the broadcast condition is not met.

  <!-- npu="950" id2 -->
  - <term>Ascend 950PR&950DT products</term>: The data type supports `torch.float16`, `torch.bfloat16`, and `torch.float32` (HF32).
  <!-- end id2 -->

- **`x2`** (`Tensor`): Required. `x2` in the matrix multiplication. The data layout can be ND. Non-contiguous Tensors are supported. When `fused_op_type` is `"relu"` or `""`, the input can have 2 to 6 dimensions. When it is `"add"` or `"mul"`, the input can be 2D (K,N) or 3D (B,K,N). In other cases, the input has 2 dimensions (K, N) and must be the same as that of `x1`. In multi-dimensional scenarios, the batch dimension must be the same when the broadcast condition is not met.

  <!-- npu="950" id3 -->
  - <term>Ascend 950PR&950DT products</term>: The data type must be the same as that of the input `x1`.
  <!-- end id3 -->

- **`*`**: Variables before it can be passed by position or assigned by using key-value pairs. Variables after it can be assigned only by using key-value pairs; variables with default values use their default values when not assigned, and variables without default values must be assigned.
- **`bias`** (`Tensor`): Optional. Bias in the matrix multiplication. The default value is None. The data layout can be ND. Non-contiguous Tensors are not supported. The input can have 2 dimensions (1, N) or 1 dimension (N,). It takes effect only when `fused_op_type` is `""`, `"relu"`, `"add"`, or `"mul"`. In other cases, use the default None.

  <!-- npu="950" id4 -->
  - <term>Ascend 950PR&950DT products</term>: The data type supports `torch.float16`, `torch.float32`, and `torch.bfloat16`. When the data type of `x1` is `torch.bfloat16`, `bias` can be `torch.float32` or `torch.bfloat16`. When the data type of `x1` is `torch.float16`, `bias` must be `torch.float16`.
  <!-- end id4 -->

- **`x3`** (`Tensor`): Optional. `x3` in the matrix multiplication. The default value is None. The data layout can be ND. Non-contiguous Tensors are supported. If `fused_op_type` is `"add"` or `"mul"`, `x3` is required. In other cases, `x3` must be None. The dimensions of `x3` must be the same as the shape of the matrix multiplication output `y` or satisfy the batch-dimension broadcast relationship. If the output has 2 dimensions (M,N), `x3` supports 2 dimensions (M,N). If the output has 3 dimensions (B,M,N), `x3` supports (B,M,N), (1,M,N), or (M,N). When the input dimensions are (1,M,N) and (M,N), `x3` must be broadcast along the batch dimension and must be the same as the output dimension. Broadcasting along M or N is not supported.

  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT products</term>: The data type must be the same as that of the input `x1`.
  <!-- end id5 -->

- **`fused_op_type`** (`str`): Required. Operator fusion type. The value can be `""` (no fusion), `"16cast32"` (no fusion, and the output is `torch.float32`; when the input is `torch.float16` or `torch.bfloat16`, the output is converted to `torch.float32`), `"add"`, `"mul"`, `"gelu_tanh"`, `"gelu_erf"`, or `"relu"`.

  <!-- npu="950" id6 -->
  - <term>Ascend 950PR&950DT products</term>: The supported values are `""`, `"16cast32"`, `"add"`, `"mul"`, `"gelu_tanh"`, `"gelu_erf"`, and `"relu"`.
  <!-- end id6 -->
  - When `fused_op_type` is `"gelu_erf"` or `"gelu_tanh"`, the data types of `x1` and `x2` must be `torch.bfloat16` or `torch.float16`.
  - When `fused_op_type` is `""` or `"relu"`, the data types of `x1` and `x2` must be `torch.bfloat16`, `torch.float16`, or `torch.float32` (HF32).
  - When `fused_op_type` is `"add"` or `"mul"`, the data types of `x1`, `x2`, and `x3` must be `torch.bfloat16`, `torch.float16`, or `torch.float32` (HF32).

## Return Values

`Tensor`

Final computation result. When `fused_op_type` is `"16cast32"`, the data type of `y` is `torch.float32`. When `fused_op_type` is not `"16cast32"`, the data type is the same as that of the input `x1`. The data layout can be ND. When `fused_op_type` is `"relu"` or `""`, the output can have 2 to 6 dimensions. When it is `"add"` or `"mul"`, the output can be 2D (M,N) or 3D (B,M,N). In other cases, the output has 2 dimensions (M, N).

## Constraints

- This API supports training and inference scenarios.
- This API supports single-operator mode and static graph mode.
- When the data type of `x1` and `x2` is `torch.float32`, HF32 must be enabled when this API is called; otherwise, it is not supported.
- When `fused_op_type` is `"add"` or `"mul"`, in BMM (3D) scenarios, `x1`, `x2`, and `y` support 3 dimensions. `x3` supports 2 or 3 dimensions. A 2D `x3` can be broadcast as a matrix for a 3D output. The batch axis of a 3D `x3` must be the same as that of `y` or be 1.

## Examples

- Single-operator call

    ```python
    import os
    import torch
    import torch_npu
    import numpy as np
    M, K, N = 128, 1, 16
    x1 = torch.randn((M, K), dtype=torch.float16)
    x2 = torch.randn((K, N), dtype=torch.float16)
    bias = None
    x3 = torch.randn((M, N), dtype=torch.float16)
    fused_op_type = "add"
    y = torch_npu.npu_fused_matmul(x1.npu(), x2.npu(), bias=bias, x3=x3.npu(), fused_op_type=fused_op_type)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import numpy as np
    import os
    import sys, getopt
    import random
    if torch.__version__ >= "2.0.0":
        from torch_npu.dynamo import torchair as tng
        from torch_npu.dynamo.torchair import CompilerConfig
        torch._dynamo.config.suppress_errors = True
    os.environ["ENABLE_ACLNN"] = "false"
    class FusedMatmulModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, x3, fused_op_type):
            out = torch_npu.npu_fused_matmul(x1, x2, bias=None, x3=x3, fused_op_type=fused_op_type)
            return out
    def executor(device):
        backend = "inductor"
        if device == "npu":
            config = CompilerConfig()
            backend = tng.get_npu_backend(compiler_config=config)
        return backend
    if __name__ == '__main__':
        argv = sys.argv[1:]
        opts, ars = getopt.getopt(argv, "hi:o:", ["npu_mode="])
        npu_mode = ""
        print("begin to run torch_stc_model")
        model = FusedMatmulModel()
        model = model.npu()
        model = torch.compile(model, backend=executor("npu"), dynamic=False, fullgraph=True)

        m = 2
        n = 3
        k = 4
        input_shape_0 = (m, k)
        input_shape_1 = (k, n)
        fused_op_type = random.choice(["", "add", "mul", "gelu_tanh", "gelu_erf", "relu"])
        fmap = torch.from_numpy(np.ones(input_shape_0).astype(np.float16))
        fmap_npu = fmap.npu()
        weight = torch.from_numpy(np.ones(input_shape_1).astype(np.float16))
        weight_npu = weight.npu()
        x3 = torch.from_numpy(np.random.rand(m, n)).to(torch.float16)
        x3_npu = x3.npu()
        if "gelu" in fused_op_type or fused_op_type == "" or fused_op_type == "relu":
            x3_npu = None
        output_data = model(fmap_npu, weight_npu, x3_npu, fused_op_type)
        print("successed!")
        print(output_data.cpu())
    ```
