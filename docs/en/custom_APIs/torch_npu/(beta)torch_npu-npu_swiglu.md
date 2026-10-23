# (beta) torch_npu.npu_swiglu

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->

## Function

- API function: Swish-gated linear unit activation function, which computes the swiglu of the Tensor `input`.

- Formulas:

     In the formulas, $x$ is the Tensor of the input parameter `input`. $dim$ is the split dimension, which is -1 by default. $A$ and $B$ are the Tensors obtained by splitting `input` along the `dim` dimension. A indicates the first half tensor, and B indicates the second half tensor.
    $$
    outputs=swiglu(x,dim=-1)=swish(A)*B=A*sigmoid(A)*B
    $$

## Prototype

```python
torch_npu.npu_swiglu(Tensor input, int dim=-1) -> (Tensor)
```

## Parameters

- **`input`** (`Tensor`): Required. Data to be computed, $x$ in the formulas. The shape can have 1 to 8 dimensions, and the shape must be divisible by 2 in the dimension corresponding to the input parameter `dim`. Non-contiguous Tensors and empty Tensors are not supported. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`.

- **`dim`** (`int`): Optional. The default value is -1. Sequence number of the dimension to be split, which splits the corresponding axis of `input` in half. The data type is `int64`, and the value range is [-input.dim(), input.dim()-1].

## Return Values

`Tensor`

Corresponds to $outputs$ in the formulas. The data type is the same as that of the computation input `input`. The data layout can be ND. Non-contiguous Tensors are not supported.

## Examples

- Single-operator call

  ```python
  import torch
  import torch_npu
  input_tensor = torch.randn(2, 32, 6, 6)
  output = torch_npu.npu_swiglu(input_tensor.npu(), dim = -1)
  ```

<!-- npu="950" id3 -->
- Graph mode call: Applicable only to <term>Ascend 950PR&950DT products</term>.

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, dim):
            return torch_npu.npu_swiglu(x, dim=dim)

    x = torch.rand((2, 32), dtype=torch.float32, device="npu")
    model = Net()
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)
    output = model(x, dim=-1)
    ```
<!-- end id3 -->
