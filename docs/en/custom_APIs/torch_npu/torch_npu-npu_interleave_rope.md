# torch\_npu.npu\_interleave\_rope

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |

## Function

- Description: Applies rotary positional encoding (RoPE) to a single input tensor `x`.
- Formulas:

    $$q = \text{reshape}(x, [B, N, S, D/2, 2]).\text{transpose}(-1, -2).\text{reshape}([B, N, S, D])$$

    $$q_{\text{embed}} = q \cdot \cos + \text{RotateHalf}(q) \cdot \sin$$

     `RotateHalf(q)` performs the following transformation on the tensor `q` along the last dimension $D$.

     1. Split the $D$ dimension evenly into two halves, where the size of each half is $D/2$.
     2. Move the second half forward and negate its values, while moving the first half backward and keeping its original values.

     Mathematical expression: Given that $q = [q_0, q_1, \ldots, q_{D/2-1}, q_{D/2}, q_{D/2+1}, \ldots, q_{D-1}]$,

     $$\text{RotateHalf}(q) = [-q_{D/2}, -q_{D/2+1}, \ldots, -q_{D-1}, q_0, q_1, \ldots, q_{D/2-1}]$$

     Example: Given that $q = [1, 2, 3, 4]$, the first half is $[1, 2]$ and the second half is $[3, 4]$. After transformation, $\text{RotateHalf}(q) = [-3, -4, 1, 2]$.

    $$\text{RotateHalf}(q)_i = \begin{cases} -q_{i+D/2}, & i < D/2 \\ q_{i-D/2}, & i \geq D/2 \end{cases}$$

## Prototype

```python
torch_npu.npu_interleave_rope(x, cos, sin) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. Tensor to be processed. This parameter must be 4D with shape `(B, N, S, D)`. The data type can be `bfloat16` or `float16`. The data layout can be ND. Non-contiguous tensors are not supported.
- **`cos`** (`Tensor`): Required. Cosine component of the RoPE rotary positional encoding. This parameter must be 4D with shape `(B, N, S, D)`, where the size of the $S$ dimension can be `1` or identical to that of `x`. The data type and data format must be identical to those of `x`. Non-contiguous tensors are not supported.
- **`sin`** (`Tensor`): Required. Sine component of the RoPE rotary positional encoding. The shape, data type, and data format must be identical to those of `cos`. Non-contiguous tensors are not supported.

## Return Values

`Tensor`

Result after applying rotary positional encoding. The shape, data type, and data format are identical to those of the input `x`. Non-contiguous tensors are not supported.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.
- The size of the $D$ dimension for `x`, `cos`, and `sin` must be equal to `64`.
- The size of the $N$ dimension for `cos` and `sin` must be equal to `1`.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    
    # Generate random data.
    x = torch.randn(32, 32, 1, 64, dtype = torch.float16)
    cos = torch.randn(32, 1, 1, 64, dtype = torch.float16)
    sin = torch.randn(32, 1, 1, 64, dtype = torch.float16)
    x_npu = x.npu()
    cos_npu = cos.npu()
    sin_npu = sin.npu()
    
    # Call the InterleaveRope operator.
    q_embed_npu = torch_npu.npu_interleave_rope(x_npu, cos_npu, sin_npu)
    ```

- Graph mode call

    ```python
    # Configure graph capture
    import torch
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    
    # Generate random data.
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
    
    # Instantiate the model.
    model = Model().npu()
    # Obtain the default backend provided by the NPU from TorchAir.
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    # Use the backend of TorchAir to call the compile API to compile the model.
    model = torch.compile(model, backend=npu_backend)
    
    # Call the InterleaveRope operator.
    q_embed_npu = model(x_npu, cos_npu, sin_npu)
    ```
