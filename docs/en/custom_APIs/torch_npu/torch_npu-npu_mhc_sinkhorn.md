# torch\_npu.npu\_mhc\_sinkhorn

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product | Supported |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## Function

- Performs Sinkhorn iterative normalization on the H<sup>res</sup>' matrix (that is, the input data of the mHC layer) in the mHC architecture to obtain the doubly stochastic matrix H<sub>res</sub>. It supports outputting the intermediate normalization results (<code>norm_out</code>) and summation results (<code>sum_out</code>) during the iterations for backward gradient computation.

- Formulas:

     1. During the first iteration (initialization):

        $$
        \begin{aligned}
        \operatorname{norm\_out}[0] &= \operatorname{softmax}(x, \dim=-1) + \varepsilon\\
        \operatorname{sum\_out}[1] &= \sum_{\dim=-2,keepdim=True} \operatorname{norm\_out}[0] + \varepsilon\\
        \operatorname{norm\_out}[1] &= \frac{\operatorname{norm\_out}[0]}{\operatorname{sum\_out}[1]}
        \end{aligned}
        $$

    2. During the i-th iteration (i = 1, 2, ..., num_iters - 1):

        $$
        \begin{aligned}
        \operatorname{sum\_out}[2i] &= \sum_{\dim=-1,keepdim=True} \operatorname{norm\_out}[2i-1] + \varepsilon\\
        \operatorname{norm\_out}[2i] &= \frac{\operatorname{norm\_out}[2i-1]}{\operatorname{sum\_out}[2i]}\\
        \operatorname{sum\_out}[2i+1] &= \sum_{\dim=-2,keepdim=True} \operatorname{norm\_out}[2i] + \varepsilon\\
        \operatorname{norm\_out}[2i+1] &= \frac{\operatorname{norm\_out}[2i]}{\operatorname{sum\_out}[2i+1]}
        \end{aligned}
        $$

    3. Final output:

        $$
        \operatorname{output} = \operatorname{norm\_out}[2 \times num\_iters - 1]
        $$

## Prototype

```python
torch_npu.npu_mhc_sinkhorn(x, eps=1e-6, num_iters=20, out_flag=0) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Tensor to be processed, representing the input data of the mHC layer in the network. The data type can be `float32`. The shape can be `[B, S, n, n]` or `[T, n, n]`. The data layout can be `ND`. Non-contiguous tensors are supported. Empty tensors are not supported. The value of `n` in the shape can only be `4`, `6`, or `8`.
- **`eps`** (`float`): Optional. Parameter used to prevent division by zero during normalization. The data type can be `float32`. The default value is `1e-6`.
- **`num_iters`** (`int`): Optional. Number of iterations. The default value is `20`. The value range is `[1, 100]`.
- **`out_flag`** (`int`): Optional. Determines whether to output intermediate results. Supported values are `0` (default) and `1`. Only the final transformation result is output when the value is `0`. `y`, `norm_out`, and `sum_out` are output when the value is `1`.

## Return Values

- **`y`** (`Tensor`): Required output. Final result of the MhcSinkhorn transformation. The data type can be `float32`. The shape can be `[B, S, n, n]` or `[T, n, n]`. The data layout can be `ND`. Non-contiguous tensors are supported. The dimensions must be the same as those of input `x`. The value of `n` in the shape can only be `4`, `6`, or `8`.
- **`norm_out`** (`Tensor`): Optional output. Normalization intermediate results during iterations. The data type can be `float32`. The shape can be `[2numIters, n, n, B, S]` or `[2numIters, n, n, T]`. The data layout can be `ND`. Non-contiguous tensors are supported. This output is valid if and only if `out_flag` is `1`. In training and inference prefill scenarios, the supported ranges of `B*S` are `[512, 65536]` and `[1, 512]`, respectively.

- **`sum_out`** (`Tensor`): Optional output. Summation intermediate results during iterations. The data type can be `float32`. The shape can be `[2numIters, n, B, S]` or `[2numIters, n, T]`. The data layout can be `ND`. Non-contiguous tensors are supported. This output is valid if and only if `out_flag` is `1`.

## Constraints

- This API can be used in inference scenarios.
- This API supports single-operator and TorchAir graph mode calls.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    device=torch.device(f'npu:0')
    torch_npu.npu.set_device(device)
    
    x_shape = [1, 128, 4 , 4]
    x = torch.rand(x_shape, dtype=torch.float32).clamp(min=1e-4)
    x_npu = x.npu()
    eps = 1e-6
    num_iters = 20
    out_flag = 0
    y = torch_npu.npu_mhc_sinkhorn(x_npu, eps=eps, num_iters=num_iters, out_flag=out_flag)
    ```

- TorchAir graph mode call (aclgraph)

    ```python
    import torch
    import torch_npu
    import torchair as tng
    config = tng.CompilerConfig()
    config.mode="reduce-overhead"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    
    device=torch.device(f'npu:0')
    torch_npu.npu.set_device(device)
    
    class MhcSinkhornModel(torch.nn.Module):
        def __init__(self):
                super().__init__()
        def forward(self, x, eps, num_iters, out_flag):
                y = torch_npu.npu_mhc_sinkhorn(x, eps=eps, num_iters=num_iters, out_flag=out_flag)
                return y
    
    x_shape = [1, 128, 4 , 4]  
    x = torch.rand(x_shape, device="npu", dtype=torch.float32)  
    eps = 1e-6
    num_iters = 20
    out_flag = 0
    mhc_sinkhorn_model = MhcSinkhornModel().npu()
    mhc_sinkhorn_model = torch.compile(mhc_sinkhorn_model, backend=npu_backend)
    y = mhc_sinkhorn_model(x, eps=eps, num_iters=num_iters, out_flag=out_flag)
    ```
