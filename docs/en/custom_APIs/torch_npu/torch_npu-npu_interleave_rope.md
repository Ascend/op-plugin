# torch\_npu.npu\_interleave\_rope

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->

## Function

- Function: performs rotary position embedding on a single input `x`.
- Formulas:

    $$q = \text{reshape}(x, [B, N, S, D/2, 2]).\text{transpose}(-1, -2).\text{reshape}([B, N, S, D])$$

    $$q_{\text{embed}} = q \cdot \cos + \text{RotateHalf}(q) \cdot \sin$$

     Where: RotateHalf\(q\) performs the following transformation on the tensor `q` on the last dimension D:

     1. Divide the D dimension equally into a front part and a back part. The length of the front part is $D/2$, and the length of the back part is also $D/2$.
     2. The elements of the back part are shifted forward as a whole and negated, and the elements of the front part are shifted backward as a whole, keeping their original values.

     Mathematical representation: Let $q = [q_0, q_1, \ldots, q_{D/2-1}, q_{D/2}, q_{D/2+1}, \ldots, q_{D-1}]$, then

     $$\text{RotateHalf}(q) = [-q_{D/2}, -q_{D/2+1}, \ldots, -q_{D-1}, q_0, q_1, \ldots, q_{D/2-1}]$$

     Example: Let $q = [1, 2, 3, 4]$. The front part is $[1, 2]$ and the back part is $[3, 4]$. After the transformation, $\text{RotateHalf}(q) = [-3, -4, 1, 2]$.

    $$\text{RotateHalf}(q)_i = \begin{cases} -q_{i+D/2}, & i < D/2 \\ q_{i-D/2}, & i \geq D/2 \end{cases}$$

## Prototype

```python
torch_npu.npu_interleave_rope(x, cos, sin) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. Tensor to be processed. This parameter must be 4D with shape `(B, N, S, D)`. The data type can be `torch.bfloat16` or `torch.float16`. The data layout can be ND. Non-contiguous tensors are not supported.

    <!-- npu="A3,910b" id4 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the data type can be `torch.bfloat16` or `torch.float16`, and non-contiguous tensors are not supported.
    <!-- end id4 -->
    <!-- npu="950" id5 -->
    - <term>Ascend 950PR&950DT products</term>: the data type can be `torch.bfloat16`, `torch.float16`, or `torch.float32`, and non-contiguous tensors are supported.
    <!-- end id5 -->

- **`cos`** (`Tensor`): Required. Cosine component of the RoPE rotary position embedding. This parameter must be 4D with shape `(B, N, S, D)`, where the size of the $S$ dimension can be `1` or identical to that of `x`. The data type and data format must be identical to those of `x`. Non-contiguous tensors are not supported.

    <!-- npu="A3,910b" id6 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the data type can be `torch.bfloat16` or `torch.float16`, and non-contiguous tensors are not supported.
    <!-- end id6 -->
    <!-- npu="950" id7 -->
    - <term>Ascend 950PR&950DT products</term>: the data type can be `torch.bfloat16`, `torch.float16`, or `torch.float32`, and non-contiguous tensors are supported.
    <!-- end id7 -->

- **`sin`** (`Tensor`): Required. Sine component of the RoPE rotary position embedding. The shape, data type, and data format must be identical to those of `cos`. Non-contiguous tensors are not supported.

    <!-- npu="A3,910b" id8 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the data type can be `torch.bfloat16` or `torch.float16`, and non-contiguous tensors are not supported.
    <!-- end id8 -->
    <!-- npu="950" id9 -->
    - <term>Ascend 950PR&950DT products</term>: the data type can be `torch.bfloat16`, `torch.float16`, or `torch.float32`, and non-contiguous tensors are supported.
    <!-- end id9 -->

## Return Values

`Tensor`

Represents the result after rotary encoding. The shape, data type, and data layout are the same as those of the input `x`. Non-contiguous tensors are not supported.

## Constraints

- This API supports only inference scenarios.
- This API supports graph mode.

<!-- npu="A3,910b" id10 -->
- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the D dimension of the inputs `x`, `cos`, and `sin` must all be equal to 64, and the N dimension of `cos` and `sin` must be equal to 1.
<!-- end id10 -->
<!-- npu="950" id11 -->
- <term>Ascend 950PR&950DT products</term>:
    - The size of the D dimension of the input tensors `x`, `cos`, and `sin` and the output tensor must be equal to 64.
    - The shapes of the input tensors `cos` and `sin` must be exactly the same, and the N dimension must be equal to 1. The S dimension can be 1 or the same as the S dimension of `x`.
<!-- end id11 -->

## Example

- Single-operator call

    ```python
    import torch
    import torch_npu
    
    # Generate random data
    x = torch.randn(32, 32, 1, 64, dtype = torch.float16)
    cos = torch.randn(32, 1, 1, 64, dtype = torch.float16)
    sin = torch.randn(32, 1, 1, 64, dtype = torch.float16)
    x_npu = x.npu()
    cos_npu = cos.npu()
    sin_npu = sin.npu()
    
    # Call the InterleaveRope operator
    q_embed_npu = torch_npu.npu_interleave_rope(x_npu, cos_npu, sin_npu)
    ```

- Graph mode call

    ```python
    # Graph mode approach
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    
    # Generate random data
    x = torch.randn(32, 32, 1, 64, dtype = torch.float16)
    cos = torch.randn(32, 1, 1, 64, dtype = torch.float16)
    sin = torch.randn(32, 1, 1, 64, dtype = torch.float16)
    x_npu = x.npu()
    cos_npu = cos.npu()
    sin_npu = sin.npu()
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x_npu, cos_npu, sin_npu):
            return torch_npu.npu_interleave_rope(x_npu, cos_npu, sin_npu)
    
    # Instantiate the model
    model = Model().npu()
    # Obtain the default backend provided by the NPU from TorchAir
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    # Use the TorchAir backend to call the compile API to compile the model
    model = torch.compile(model, backend=npu_backend)
    
    # Call the InterleaveRope operator
    q_embed_npu = model(x_npu, cos_npu, sin_npu)
    ```
