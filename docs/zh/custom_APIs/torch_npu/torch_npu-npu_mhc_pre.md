# torch\_npu.npu\_mhc\_pre

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Atlas 350 加速卡</term> | √ |

## 功能说明

- API功能：基于一系列计算得到mHC\(Manifold-Constrained Hyper-Connections\)架构中hidden层的H<sup>res</sup>和H<sup>post</sup>投影矩阵，以及Atten层或MLP层的输入矩阵h<sup>in</sup>。
- 计算公式：

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
其中：
$$
\operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+norm\_eps}
$$
   
## 函数原型

```python
torch_npu.npu_mhc_pre(x, phi, alpha, bias, *, gamma=None, norm_eps=1e-6, hc_eps=1e-6, out_flag=0) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **x**（`Tensor`）：必选参数，待计算的数据，表示网络中mHC层的输入数据，数据类型支持`bfloat16`、`float16`，shape为\(B, S, n, D\)或\(T, n, D\)，数据格式支持ND，支持非连续Tensor，不支持空Tensor。
- **phi**（`Tensor`）：必选参数，mHC的参数矩阵，顺序是W\_pre\(n, nD\)、W\_post\(n, nD\)、W\_res\(n<sup>2</sup>, nD\)，数据类型为`float32`，shape为\(n<sup>2</sup>+2n, nD\)，数据格式支持ND，支持非连续Tensor，不支持空Tensor。
- **alpha**（`Tensor`）：必选参数，mHC的缩放参数，顺序是alpha\_pre、alpha\_post、alpha\_res，数据类型为`float32`，shape为\(3\)，不支持空Tensor。
- **bias**（`Tensor`）：必选参数，mHC层的bias参数，数据类型为`float32`，shape为\(n<sup>2</sup>+2n\)，不支持空Tensor。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **gamma**（`Tensor`）：可选参数，表示进行RmsNorm的缩放因子，数据类型为`float32`，shape为\(n, D\)，数据格式支持ND，支持非连续Tensor。
- **norm\_eps**（`float`）：可选参数，RmsNorm的防除零参数，默认值是1e-6，数据类型为`float32`。
- **hc\_eps**（`float`）：可选参数，H<sub>pre</sub>的sigmoid后的eps参数，默认值是1e-6，数据类型支持`float32`。
- **out\_flag**（`int`）：可选参数，表示是否输出h\_mix/inv\_rms/h\_pre，默认为0表示不输出，为1表示全输出。

## 返回值说明

- **h\_in**（`Tensor`）：输出的h\_in作为Atten/MLP层的输入，数据类型为`bfloat16`、`float16`，shape为\(B, S, D\)或\(T, D\)，数据格式支持ND。
- **h\_post**（`Tensor`）：输出的mHC的h\_post变换矩阵，数据类型为`float32`，shape为\(B, S, D\)或\(T,  D\)，数据格式支持ND。
- **h\_res**（`Tensor`）：输出的mHC的h\_res变换矩阵（未做sinkhorn变换），数据类型为`float32`，shape为\(B, S, n, n\)或\(T, n, n\)，数据格式支持ND。
- **inv\_rms**（`Tensor`）：可选输出，RmsNorm计算得到的1/r，数据类型为`float32`，shape为\(B, S\)或\(T\)，数据格式支持ND。
- **h\_mix**（`Tensor`）：可选输出，x与phi矩阵乘的结果，数据类型为`float32`，shape为\(B, S, n<sup>2</sup>+2n\)或\(T,  n<sup>2</sup>+2n\)，数据格式支持ND。
- **h\_pre**（`Tensor`）：可选输出，做完sigmoid计算之后的h\_pre矩阵，数据类型为`float32`，shape为\(B, S, n\)或\(T,  n\)，数据格式支持ND。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用。
- Shape规格约束：
  - n：目前支持4、6、8。
  - D：支持1\~16384，需满足D为16对齐。

## 调用示例

- 单算子模式调用

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

- TorchAir图模式调用（aclgraph）

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
    # 配置图模式config
    config = torchair.CompilerConfig()
    # 配置图执行模式，aclgraph模式为reduce-overhead
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
