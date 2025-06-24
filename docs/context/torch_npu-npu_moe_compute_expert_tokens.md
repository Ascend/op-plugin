# torch_npu.npu_moe_compute_expert_tokens

## 功能说明

- 算子功能：MoE（Mixture of Experts，混合专家模型）计算中，通过二分查找的方式查找每个专家处理的最后一行的位置。
- 计算公式：

    ![](figures/zh-cn_formulaimage_0000001915130445.png)

## 函数原型

```
torch_npu.npu_moe_compute_expert_tokens(Tensor sorted_expert_for_source_row, int num_expert) -> Tensor
```

## 参数说明

- sorted_expert_for_source_row：Tensor类型，必选参数，经过专家处理过的结果，要求是一个1D的Tensor，数据类型支持int32，数据格式要求为ND。shape大小需要小于2147483647。

- num_expert：int类型，必选参数，总专家数。

## 输出说明

expertTokens：Tensor类型，公式中的输出，要求的是一个1D的Tensor，数据类型与sorted_expert_for_source_row保持一致。

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
    sorted_experts = torch.tensor([3,3,4,5,6,7], dtype=torch.int32)
    num_experts = 5
    output = torch_npu.npu_moe_compute_expert_tokens(sorted_experts.npu(), num_experts)
    ```

- 图模式调用

    ```python
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    class GMMModel(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, sorted_experts, num_experts):
            return torch_npu.npu_moe_compute_expert_tokens(sorted_experts, num_experts)
    def main():
        sorted_experts = torch.tensor([3,3,4,5,6,7], dtype=torch.int32)
        num_experts = 5
        model = GMMModel().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        custom_output = model(sorted_experts, num_experts)
    if __name__ == '__main__':
        main()
    ```

