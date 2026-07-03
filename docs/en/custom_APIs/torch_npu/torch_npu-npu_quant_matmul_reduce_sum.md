# torch_npu.npu_quant_matmul_reduce_sum

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>     |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |

## Function

- Description: Performs quantized grouped matrix multiplication, sums up the matrix multiplication results of all groups, and outputs the result.

- Formula:

$$
out = \sum_{i=0}^{batch}(x1_i @ x2_i) * x1Scale_i * x2Scale
$$

## Prototype

```python
torch_npu.npu_quant_matmul_reduce_sum(x1, x2, *, x1_scale=None, x2_scale=None) -> Tensor
```

## Parameters

- **`x1`** (`Tensor`): Required. The data type can be `int8`. The data layout can be ND. This parameter must be 3D with shape `(batch, m, k)`.

- **`x2`** (`Tensor`): Required. The data type can be `int8`. The data layout can only be NZ. This parameter must be 3D with shape `(batch, k, n)`. If the input data layout is ND, `x2` can be converted from ND layout to NZ layout through `x2 = torch_npu.npu_format_cast(x2.contiguous(), 29)`, where `29` is the enumeration value of the NZ layout.

- **`x1_scale`** (`Tensor`): Required keyword parameter, $x1Scale$ in the formula. The data type can be `float32`. The data layout can be ND. This parameter must be 2D with shape `(batch, m)`. During actual computation, `x1_scale` is broadcast to `(batch, m, n)`.

- **`x2_scale`** (`Tensor`): Required keyword parameter, $x2Scale$ in the formula. The data type can be `bfloat16`. The data layout can be ND. This parameter must be 1D with shape `(n,)`. During actual computation, `x2_scale` is broadcast to `(batch, m, n)`.

## Return Values

`Tensor`

Computation result of the operator, $out$ in the formula. The output data type is `bfloat16`. The data layout can be ND. This parameter must be 2D with shape `(m, n)`.

## Constraints

- This API can be used in inference scenarios.
- This API supports static graph mode.
- The input parameters `x1`, `x2`, `x1_scale`, and `x2_scale` must not be empty tensors.
- The following table describes the supported input and output data type combinations.

  | x1   | x2   | x1_scale | x2_scale  | out      |
  |------|------|---------|----------|----------|
  | int8 | int8 | float32 | bfloat16 | bfloat16 |

## Examples

- Single-operator call

  ```python
  import torch
  import torch_npu

  b,m,k,n = (2,3,4,5)
  x1 = torch.ones((b, m, k), dtype=torch.int8).npu()
  x2_nd = torch.ones((b, k, n), dtype=torch.int8).npu()
  x2 = torch_npu.npu_format_cast(x2_nd.contiguous(), 29)
  x1_scale = torch.ones((b, m), dtype=torch.float32).npu()
  x2_scale = torch.ones((n,), dtype=torch.bfloat16).npu()
  y = torch_npu.npu_quant_matmul_reduce_sum(x1, x2, x1_scale=x1_scale, x2_scale=x2_scale)
  ```

- Graph mode call

  ```python
  import torch
  import torch_npu
  import torchair as tng
  from torchair.ge_concrete_graph import ge_apis as ge
  from torchair.configs.compiler_config import CompilerConfig
  import logging
  from torchair.core.utils import logger

  logger.setLevel(logging.DEBUG)
  import os
  import numpy as np

  # ENABLE_ACLNN specifies whether to use ACLNN. Valid values: true (uses ACLNN execution) or false (uses online compilation).
  os.environ["ENABLE_ACLNN"] = "false"
  config = CompilerConfig()
  npu_backend = tng.get_npu_backend(compiler_config=config)

  class MyModel(torch.nn.Module):
      def __init__(self):
          super().__init__()
 
      def forward(self, x1, x2, scale, pertoken_scale):
          return torch_npu.npu_quant_matmul_reduce_sum(x1, x2, x1_scale=pertoken_scale, x2_scale=scale)

  cpu_model = MyModel()
  model = cpu_model.npu()
  model = torch.compile(model, backend=npu_backend, dynamic=False)

  b,m,k,n = (2,3,4,5)
  x1 = torch.ones((b, m, k), dtype=torch.int8).npu()
  x2_nd = torch.ones((b, k, n), dtype=torch.int8).npu()
  x2 = torch_npu.npu_format_cast(x2_nd.contiguous(), 29)
  pertoken_scale = torch.ones((b, m), dtype=torch.float32).npu()
  scale = torch.ones((n,), dtype=torch.bfloat16).npu()
  npu_out = model(x1, x2, scale, pertoken_scale)
  print(npu_out)
  ```
