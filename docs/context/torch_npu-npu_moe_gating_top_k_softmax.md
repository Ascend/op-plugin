# torch_npu.npu_moe_gating_top_k_softmax

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>  | √   |

## 功能说明

- API功能：MoE计算中，对输入`x`做Softmax计算，再做topk操作。
- 计算公式：
$$
softmaxOut = softmax(x, axis = -1) \\
yOut, expertIdxOut = topK(softmaxOut, k = k) \\
rowIdxRange = orange(expertIdxOut.shape[0] * expertIdxOut.shape[1])\\
rowIdxOut = rowIdxRange.reshape([expertIdxOut.shape[1], expertIdxOut.shape[0]]).transpose(1, 0)
$$

## 函数原型

```
torch_npu.npu_moe_gating_top_k_softmax(x, finished=None, k=1) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：必选参数，公式中的$x$，表示待计算的输入，要求为2维/3维张量，数据类型支持`float16`、`bfloat16`、`float32`，数据格式要求为$ND$。
- **finished** (`Tensor`)：可选参数，表示输入中需要参与计算的行，要求为2维/3维张量，数据类型支持`bool`，shape为gating_shape[:-1]，数据格式要求为$ND$。
- **k** (`int`)：可选参数，公式中的$k$，表示topk的k值，大小为0<k<=x的-1轴大小，k<=1024。

## 返回值说明

- **y** (`Tensor`)：对应公式中的$yOut$，对x做softmax后取的topk值，数据维度支持2维/3维，数据类型与`x`需要保持一致，其非-1轴要求与`x`的对应轴大小一致，其-1轴要求其大小同`k`值。数据格式要求为$ND$。
- **expert_idx** (`Tensor`)：对应公式中的$expertIdxOut$，对`x`做softmax后取topk值的索引，即专家的序号。shape要求与`y`一致，数据类型支持`int32`，数据格式要求为$ND$。
- **row_idx** (`Tensor`)：对应公式中的$rowIdxOut$，表示输出的行位置对应输入的行位置，shape要求与`y`一致，数据类型支持`int32`，数据格式要求为$ND$。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式（PyTorch 2.1.0版本）。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    x = torch.rand((3, 3), dtype=torch.float32).to("npu")
    finished = torch.randint(2, size=(3,), dtype=torch.bool).to("npu")
    y, expert_idx, row_idx = torch_npu.npu_moe_gating_top_k_softmax(x, finished, k=2)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    torch_npu.npu.set_compile_mode(jit_compile=True)
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    device=torch.device(f'npu:0')
    torch_npu.npu.set_device(device)
    class MoeGatingTopkSoftmaxModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, finish, k):
            res = torch_npu.npu_moe_gating_top_k_softmax(x, finish, k)
            return res
    x = torch.randn((2, 4, 6),device='npu',dtype=torch.float16).npu()
    moe_gating_topk_softmax_model = MoeGatingTopkSoftmaxModel().npu()
    moe_gating_topk_softmax_model = torch.compile(moe_gating_topk_softmax_model, backend=npu_backend, dynamic=True)
    res = moe_gating_topk_softmax_model(x, None, 2)
    print(res)
    ```

