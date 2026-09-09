# torch\_npu.npu\_mhc\_pre

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product | Supported |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## Function

- Obtains the $H^{res}$ and $H^{post}$ projection matrices of the hidden layer in the Manifold-Constrained Hyper-Connections (mHC) architecture, as well as the input matrix $h^{in}$ of the Attention or MLP layer.

- Formulas:

$$
\begin{aligned}
\vec{x^{'}_{l}} &=\operatorname{RmsNorm}(\vec{x_{l}})\\
H^{pre}_l &= \alpha^{pre}_{l} ·(\vec{x^{'}_{l}}\varphi^{pre}_{l}) + b^{pre}_{l}\\
H^{post}_l &= \alpha^{post}_{l} ·(\vec{x^{'}_{l}}\varphi^{post}_{l}) + b^{post}_{l}\\
H^{res}_l &= \alpha^{res}_{l} ·(\vec{x^{'}_{l}}\varphi^{res}_{l}) + b^{res}_{l}\\
H^{pre}_l &= \sigma (H^{pre}_{l}) + hc\_eps\\
H^{post}_l &= 2\sigma (H^{post}_{l})\\
h_{in} &=\vec{x_{l}}H^{pre}_l
\end{aligned}
$$
where
$$
\operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+norm\_eps}
$$
   
## Prototype

```python
torch_npu.npu_mhc_pre(x, phi, alpha, bias, *, gamma=None, norm_eps=1e-6, hc_eps=1e-6, out_flag=0) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Data to be processed, representing the input data of the mHC layer in the network. The data type can be `bfloat16` or `float16`. The shape can be `(B, S, n, D)` or `(T, n, D)`. The data layout can be `ND`. Non-contiguous tensors are supported. Empty tensors are not supported.
- **`phi`** (`Tensor`): Required. Parameter matrix of mHC, in the order of `W_pre` `(n, nD)`, `W_post` `(n, nD)`, and `W_res` `(n<sup>2</sup>, nD)`. The data type is `float32`. The shape is `(n<sup>2</sup>+2n, nD)`. The data layout can be `ND`. Non-contiguous tensors are supported. Empty tensors are not supported.
- **`alpha`** (`Tensor`): Required. Scaling parameters of mHC, in the order of `alpha_pre`, `alpha_post`, and `alpha_res`. The data type is `float32`. The shape is `(3)`. Empty tensors are not supported.
- **`bias`** (`Tensor`): Required. Bias parameter of the mHC layer. The data type is `float32`. The shape is `(n<sup>2</sup>+2n)`. Empty tensors are not supported.
- **`*`**: Position delimiter. Variables before this delimiter are position-dependent and must be passed in order. Variables after this delimiter are optional keyword arguments and must be assigned using key-value pairs. If not specified, their default values are used.
- **`gamma`** (`Tensor`): Optional. Scaling factor for RmsNorm. The data type is `float32`. The shape is `(n, D)`. The data layout can be `ND`. Non-contiguous tensors are supported.
- **`norm_eps`** (`float`): Optional. Zero-division prevention parameter for RmsNorm. The default value is `1e-6`. The data type is `float32`.
- **`hc_eps`** (`float`): Optional. Epsilon parameter applied after the sigmoid operation on H<sub>pre</sub>. The default value is `1e-6`. The data type can be `float32`.
- **`out_flag`** (`int`): Optional. Specifies whether to output `h_mix`, `inv_rms`, and `h_pre`. The default value is `0`, indicating that these outputs are not generated. `1` indicates that all of these outputs are generated.

## Return Values

- **`h_in`** (`Tensor`): Output `h_in` serving as the input to the Attention/MLP layer. The data type can be `bfloat16` or `float16`. The shape can be `(B, S, D)` or `(T, D)`. The data layout can be `ND`.
- **`h_post`** (`Tensor`): Output mHC `h_post` transformation matrix. The data type is `float32`. The shape can be `(B, S, D)` or `(T, D)`. The data layout can be `ND`.
- **`h_res`** (`Tensor`): Output mHC `h_res` transformation matrix (without the Sinkhorn transformation). The data type is `float32`. The shape can be `(B, S, n, n)` or `(T, n, n)`. The data layout can be `ND`.
- **`inv_rms`** (`Tensor`): Optional output. `1/r` calculated by RmsNorm. The data type is `float32`. The shape can be `(B, S)` or `(T)`. The data layout can be `ND`.
- **`h_mix`** (`Tensor`): Optional output. Result of multiplying `x` by the `phi` matrix. The data type is `float32`. The shape can be `(B, S, n<sup>2</sup>+2n)` or `(T, n<sup>2</sup>+2n)`. The data layout can be `ND`.
- **`h_pre`** (`Tensor`): Optional output. `h_pre` matrix after applying the sigmoid operation. The data type is `float32`. The shape can be `(B, S, n)` or `(T, n)`. The data layout can be `ND`.

## Constraints

- This API can be used in inference scenarios.
- This API supports single-operator and TorchAir graph mode calls.
- Shape constraints:
  - `n`: Valid values: `4`, `6`, and `8`.
  - `D`: The value ranges from `1` to `16384` and must be a multiple of `16`.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    import numpy as np
    T=1024
    n=4
    D=2560
    x = torch.randn(T, n, D, dtype=torch.bfloat16).npu()
    phi = torch.randn(n * n + 2 * n, n * D, dtype=torch.float32).npu()
    alpha = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).npu()
    bias_pre = torch.full((n,), 0.01, dtype=torch.float32)
    bias_post = torch.full((n,), 0.01, dtype=torch.float32)
    bias_res = torch.full((n, n), 0.01, dtype=torch.float32)
    bias = torch.cat([bias_pre, bias_post, bias_res.reshape(-1)], dim=0).npu()
    gamma = torch.randn(n, D, dtype=torch.float32).npu()
    out = torch_npu.npu_mhc_pre(x, phi, alpha, bias, gamma=gamma, out_flag=1)
    names = ["h_in", "h_post", "h_comb_before", "inv_rms", "h_mix", "h_pre"]
    for name, ele in zip(names, out):
        print(f"\n{name=}")
        print(ele.float().cpu())
    ```

- TorchAir graph mode call (aclgraph)

    ```python
    import torch
    import torch_npu
    import torchair
    import logging
    import os
    import warnings
    from torchair.core.utils import logger
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logger.setLevel(logging.DEBUG)
    os.environ["ENABLE_ACLNN"] = "false"
    # Configure graph mode
    config = torchair.CompilerConfig()
    # Configure the graph execution mode. The aclgraph mode is "reduce-overhead".
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, phi, alpha, bias, gamma):
            return torch_npu.npu_mhc_pre(x.npu(), phi.npu(), alpha.npu(), bias.npu(), gamma=gamma.npu(), out_flag=1)
    if __name__ == "__main__":
        T = 1024
        n = 8
        D = 5120
        x = torch.randn(T, n, D, dtype=torch.bfloat16).npu()
        phi = torch.randn(n * n + 2 * n, n * D, dtype=torch.float32).npu()
        alpha = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).npu()
        bias_pre = torch.full((n,), 0.01, dtype=torch.float32)
        bias_post = torch.full((n,), 0.01, dtype=torch.float32)
        bias_res = torch.full((n, n), 0.01, dtype=torch.float32)
        bias = torch.cat([bias_pre, bias_post, bias_res.reshape(-1)], dim=0).npu()
        gamma = torch.randn(n, D, dtype=torch.float32).npu()
        model = MyModel()
        model = model.npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        out = model(x, phi, alpha, bias, gamma)
        names = ["h_in", "h_post", "h_comb_before", "inv_rms", "h_mix", "h_pre"]
        for name, ele in zip(names, out):
            print(f"\n{name=}")
            print(ele.float().cpu())
    ```
