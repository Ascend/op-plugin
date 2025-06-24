# torch_npu.npu_moe_gating_top_k_softmax

## 功能说明

MoE计算中，对输入x做Softmax计算，再做topk操作。

## 函数原型

```
torch_npu.npu_moe_gating_top_k_softmax(Tensor x, Tensor? finished=None, int k=1) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- x：Tensor类型，必选输入，表示待计算的输入要求是一个2D/3D的Tensor，数据类型支持float16、bfloat16、float32，数据格式要求为ND。
- finished：Tensor类型，可选输入，表示输入中需要参与计算的行，要求是一个1D/2D的Tensor，数据类型支持bool，shape为gating_shape[:-1]，数据格式要求为ND。
- k：Host侧的int类型，表示topk的k值，大小为0<k<=x的-1轴大小，k<=1024。

## 输出说明

- y：Tensor类型，对x做softmax后取的topk值，要求是一个2D/3D的Tensor，数据类型与x需要保持一致，其非-1轴要求与x的对应轴大小一致，其-1轴要求其大小同k值。数据格式要求为ND。
- expert_idx：Tensor类型，对x做softmax后取topk值的索引，即专家的序号。shape要求与y一致，数据类型支持int32，数据格式要求为ND。

- row_idx：Tensor类型，指示每个位置对应的原始行位置，shape要求与y一致，数据类型支持int32，数据格式要求为ND。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式（PyTorch 2.1版本）。

## 支持的型号

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> 
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

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

