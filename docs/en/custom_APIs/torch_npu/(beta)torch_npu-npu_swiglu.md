# (beta) torch_npu.npu_swiglu

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 training products</term>| √   |

## Function

- Description: Computes the Swish Gated Linear Unit (SwiGLU) activation function for the input tensor `input`, implementing the SwiGLU computation logic.

- Formula:

     $x$ is the tensor corresponding to the input parameter `input`. $dim$ is the split dimension and defaults to `-1`. $A$ and $B$ are tensors obtained by splitting `input` along dimension `dim`. $A$ represents the first half of the tensor, and $B$ represents the second half of the tensor.
    $$
    outputs=swiglu(x,dim=-1)=swish(A)*B=A*sigmoid(A)*B
    $$

## Prototype

```python
torch_npu.npu_swiglu(Tensor input, int dim=-1) -> (Tensor)
```

## Parameters

**`input`** (`Tensor`): Required. Input data to be computed, $x$ in the formula. The shape can have 1 to 8 dimensions and must be divisible by 2 along the dimension corresponding to the input parameter `dim`. Non-contiguous tensors and empty tensors are not supported. The data type can be `float32`, `float16`, or `bfloat16`.

**`dim`** (`int`): Optional. Sequence number of the dimension to be split. The corresponding axis of `input` is split in half. The default value is `-1`. The value range is `[–input.dim(), input.dim()–1]`.

## Return Values

`Tensor`

$outputs$ in the formula. The data type must be identical to that of the computation input `input`. Non-contiguous tensors are not supported.

## Example

```python
import torch
import torch_npu
input_tensor = torch.randn(2, 32, 6, 6)
output = torch_npu.npu_swiglu(input_tensor.npu(), dim = -1)
```
