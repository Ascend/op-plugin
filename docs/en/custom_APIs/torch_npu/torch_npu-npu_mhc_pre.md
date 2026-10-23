# torch\_npu.npu\_mhc\_pre

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- Function: obtains, through a series of computations, the $H^{res}$ and $H^{post}$ projection matrices of the hidden layer and the input matrix h<sup>in</sup> of the Atten or MLP layer in the mHC (Manifold-Constrained Hyper-Connections) architecture.
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
Where:
$$
\operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+norm\_eps}
$$

## Prototype

```python
torch_npu.npu_mhc_pre(x, phi, alpha, bias, *, gamma=None, norm_eps=1e-6, hc_eps=1e-6, out_flag=0, inner_precise=0) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Data to be processed, representing the input data of the mHC layer in the network. The data type can be `torch.bfloat16` or `torch.float16`. The shape can be `(B, S, n, D)` or `(T, n, D)`. The data layout can be `ND`. Non-contiguous tensors are supported. Empty tensors are not supported.
- **`phi`** (`Tensor`): Required. Parameter matrix of mHC, in the order of `W_pre` `(n, nD)`, `W_post` `(n, nD)`, and `W_res` `(n<sup>2</sup>, nD)`. The data type is `torch.float32`. The shape is `(n<sup>2</sup>+2n, nD)`. The data layout can be `ND`. Non-contiguous tensors are supported. Empty tensors are not supported.
- **`alpha`** (`Tensor`): Required. Scaling parameters of mHC, in the order of `alpha_pre`, `alpha_post`, and `alpha_res`. The data type is `torch.float32`. The shape is `(3)`. Empty tensors are not supported.
- **`bias`** (`Tensor`): Required. Bias parameter of the mHC layer. The data type is `torch.float32`. The shape is `(n<sup>2</sup>+2n)`. Empty tensors are not supported.
- \*: Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`gamma`** (`Tensor`): Optional. Scaling factor for RmsNorm. The data type is `torch.float32`. The shape is `(n, D)`. The data layout can be `ND`. Non-contiguous tensors are supported.
- **`norm_eps`** (`float`): Optional. RmsNorm division-by-zero prevention parameter. The default value is 1e-6 and the data type is `float32`.
- **`hc_eps`** (`float`): Optional. Eps parameter after the sigmoid of H<sub>pre</sub>. The default value is 1e-6 and the data type can be `float32`.
- **`out_flag`** (`int`): Optional. Specifies whether to output h_mix/inv_rms/h_pre. The default value 0 indicates no output, and 1 indicates that all are output.
- **`inner_precise`** (`int`): Optional. Computation mode of the MhcPre operator. The default value is 0. 0 indicates that the computation is performed in the Cube using the `torch.float32` mode; 1 indicates that the computation is performed in the Cube using the HF32 mode.

## Return Values

- **`h_in`** (`Tensor`): The output h_in serves as the input of the Atten/MLP layer. The data type is `torch.bfloat16` or `torch.float16`, the shape is (B, S, D) or (T, D), and the data layout can be ND.
- **`h_post`** (`Tensor`): Output h_post transformation matrix of mHC. The data type is `torch.float32`, the shape is (B, S, D) or (T, D), and the data layout can be ND.
- **`h_res`** (`Tensor`): Output h_res transformation matrix of mHC (without the Sinkhorn transformation). The data type is `torch.float32`, the shape is (B, S, n, n) or (T, n, n), and the data layout can be ND.
- **`inv_rms`** (`Tensor`): Optional output. 1/r computed by RmsNorm. The data type is `torch.float32`, the shape is (B, S) or (T), and the data layout can be ND.
- **`h_mix`** (`Tensor`): Optional output. Result of the matrix multiplication of x and phi. The data type is `torch.float32`, the shape is (B, S, n<sup>2</sup>+2n), (T, n<sup>2</sup>+2n), (B, S, 2n), or (T, 2n), and the data layout can be ND.
- **`h_pre`** (`Tensor`): Optional output. h_pre matrix after the sigmoid computation. The data type is `torch.float32`, the shape is (B, S, n) or (T, n), and the data layout can be ND.

## Constraints

- This API can be used in inference scenarios.
- This API supports single-operator mode and TorchAir graph mode calls.
- Shape constraints:
  - n: 4, 6, and 8 are supported currently.
  - D: 1 to 16384 is supported, and D must be 16-aligned.
- When alpha=[3], the h_res output is supported and the following conditions must be met: input phi=(n<sup>2</sup>+2n, nD) and bias=(n<sup>2</sup>+2n); output h_mix=(B, S, n<sup>2</sup>+2n) or (T, n<sup>2</sup>+2n). When alpha=[2], the h_res output is 0 and the following conditions must be met: input phi=(2n, nD) and bias=(2n); output h_mix=(B, S, 2n) or (T, 2n).
- The optional outputs inv_rms, h_mix, and h_pre are mutually dependent: they must be output together or not output at all. Returning only some of them is not supported.
  - inner_precise: 0 and 1 are supported currently.

- Deterministic computation
  - Deterministic computation is supported by default.
- Batch consistency
  - Batch consistency is not supported by default. It can be enabled using the PyTorch switch ([torch_npu.npu.set_deterministic_level](../torch_npu-npu/torch_npu-npu-set_deterministic_level.md)).

## Example

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
    # Configure the graph mode config
    config = torchair.CompilerConfig()
    # Configure the graph execution mode. The aclgraph mode is reduce-overhead.
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
