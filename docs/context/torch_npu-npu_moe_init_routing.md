# torch_npu.npu_moe_init_routing

## 功能说明

- 算子功能：MoE的routing计算，根据[torch_npu.npu_moe_gating_top_k_softmax](torch_npu-npu_moe_gating_top_k_softmax.md)的计算结果做routing处理。
- 计算公式为：

    ![](figures/zh-cn_formulaimage_0000001855915460.png)

    ![](figures/zh-cn_formulaimage_0000001902036405.png)

    ![](figures/zh-cn_formulaimage_0000001855916732.png)

## 函数原型

```
torch_npu.npu_moe_init_routing(Tensor x, Tensor row_idx, Tensor expert_idx, int active_num) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- x：tensor类型，必选输入，MOE的输入即token特征输入，要求为一个2D的tensor，shape为(NUM_ROWS, H)。数据类型支持float16、bfloat16、float32，数据格式要求为ND。shape大小需要小于2^24。
- row_idx：tensor类型，必选输入，指示每个位置对应的原始行位置，shape要求与expert_idx一致。数据类型支持int32，数据格式要求为ND。
- expert_idx：tensor类型，必选输入，[torch_npu.npu_moe_gating_top_k_softmax](torch_npu-npu_moe_gating_top_k_softmax.md)的输出每一行特征对应的K个处理专家，要求是一个2D的shape (NUM_ROWS, K)，数据类型支持int32，数据格式要求为ND。

- active_num：int类型，表示总的最大处理row数，输出expanded_x只有这么多行是有效的。

## 输出说明

- expanded_x：tensor类型，根据expert_idx进行扩展过的特征，要求是一个2D的tensor，shape (min(NUM_ROWS, active_num) \* k, H)。数据类型同x，数据格式要求为ND。
- expanded_row_idx：tensor类型，expanded_x和x的映射关系，要求是一个1D的tensor，shape为(NUM_ROWS\*K, )，数据类型支持int32，数据格式要求为ND。
- expanded_expert_idx：tensor类型，输出expert_idx排序后的结果。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式（PyTorch 2.1版本）。

## 支持的型号

<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> 

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    x = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2],[0.3, 0.3, 0.3, 0.3]], dtype=torch.float32).to("npu")
    row_idx = torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.int32).to("npu")
    expert_idx = torch.tensor([[1, 2], [0, 1], [0, 2]], dtype=torch.int32).to("npu")
    active_num = 3
    expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(x, row_idx, expert_idx, active_num)
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
    
    class MoeInitRoutingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, x, row_idx, expert_idx, active_num):
            expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(x, row_idx, expert_idx, active_num=active_num)
            return expanded_x, expanded_row_idx, expanded_expert_idx
    
    x = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2],[0.3, 0.3, 0.3, 0.3]], dtype=torch.float32).to("npu")
    row_idx = torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.int32).to("npu")
    expert_idx = torch.tensor([[1, 2], [0, 1], [0, 2]], dtype=torch.int32).to("npu")
    active_num = 3
    
    moe_init_routing_model = MoeInitRoutingModel().npu()
    moe_init_routing_model = torch.compile(moe_init_routing_model, backend=npu_backend, dynamic=True)
    expanded_x, expanded_row_idx, expanded_expert_idx = moe_init_routing_model(x, row_idx, expert_idx, active_num=active_num)
    print(expanded_x)
    print(expanded_row_idx)
    print(expanded_expert_idx)
    ```

