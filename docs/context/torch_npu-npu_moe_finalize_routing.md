# torch_npu.npu_moe_finalize_routing

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>  | √   |

## 功能说明

- API功能：在MoE计算的最后，合并MoE FFN(Feedforward Neural Network)的输出结果。

- 计算公式：

    $$
    expertid = exportForSourceRow[i,k] \\
    out(i,j) = skip1_{i,j} + skip2_{i,j} + \sum^K_{k=0}(scales_{i,k} * (expandPermutedROWs_{expandedSrcRowToDstRow\_i+k*num\_rows, j}+ bias_{expertid,j}))
    $$

## 函数原型

```
torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2, bias, scales, expanded_src_to_dst_row, export_for_source_row, drop_pad_mode=0) -> Tensor
```

## 参数说明
>**说明：**<br>
>shape中的符号说明：
>-   $NUM\_ROWS$：为行数。
>-   $K$：表示从总的专家$E$中选出$K$个专家。$E$表示专家数，$E$需要大于等于$K$。
>-   $H$：表示每个token序列长度，为列数。

- **expanded_permuted_rows** (`Tensor`)：必选参数，对应公式中的$expandPermutedROWs$，经过专家处理过的结果，要求为一个2维张量，数据类型支持`float16`、`bfloat16`、`float32`，数据格式要求为$ND$。shape支持$（NUM\_ROWS * K, H）$。
- **skip1** (`Tensor`)：必选参数，允许为None，对应公式中的$skip1$，求和的输入参数1，要求为一个2维张量，数据类型要求与`expanded_permuted_rows`一致，shape要求与输出`out`的shape一致。
- **skip2** (`Tensor`)：必选参数，允许为None，对应公式中的$skip2$，求和的输入参数2，要求为一个2维张量，数据类型要求与`expanded_permuted_rows`一致，shape要求与输出`out`的shape一致。`skip2`参数为`None`时，`skip1`参数必须也为`None`。
- **bias** (`Tensor`)：必选参数，允许为None，对应公式中的$bias$，专家的偏差，要求为一个2维张量，数据类型要求与`expanded_permuted_rows`一致。shape支持$（E，H）$。
- **scales** (`Tensor`)：必选参数，允许为None，对应公式中的$scales$，专家的权重，要求为一个2维张量，数据类型要求与`expanded_permuted_rows`一致，shape支持（$NUM\_ROWS，K）$。
- **expanded_src_to_dst_row** (`Tensor`)：必选参数，对应公式中的$expandedSrcRowToDstRow$，保存每个专家处理结果的索引，要求为一个1维为张量，数据类型支持`int32`。shape支持$（NUM\_ROWS * K）$，`drop_pad_mode`参数为0时，Tensor的取值范围是$[0, NUM\_ROWS * K-1]$。
- **export_for_source_row** (`Tensor`)：必选参数，允许为None，公式中的$exportForSourceRow$，每行处理的专家号，要求为一个2维张量，数据类型支持`int32`。shape支持$（NUM\_ROWS，K）$，Tensor的取值范围是[0,E-1]。
- **drop_pad_mode** (`int`)：可选参数，表示是否支持丢弃模式，取值范围为0，默认值为`0`。


## 返回值说明
`Tensor`
输出参数`out`,代表最后MoE FFN合并的输出结果。数据维度支持2维，shape支持$（NUM_ROWS, H）$。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式（PyTorch 2.1.0版本）。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> expert_num = 16
    >>> token_len = 10
    >>> top_k = 4
    >>> num_rows = 50
    >>> device = torch.device('npu')
    >>> dtype = torch.float32
    >>>
    >>> expanded_permuted_rows = torch.randn((num_rows * top_k, token_len), device=device, dtype=dtype)
    >>> skip1 = torch.randn((num_rows, token_len), device=device, dtype=dtype)
    >>> skip2_optional = torch.randn((num_rows, token_len), device=device, dtype=dtype)
    >>> bias = torch.randn((expert_num, token_len), device=device, dtype=dtype)
    >>> scales = torch.randn((num_rows, top_k), device=device, dtype=dtype)
    >>> expert_for_source_row = torch.randint(low=0, high=expert_num, size=(num_rows, top_k), device=device, dtype=torch.int32)
    >>> expanded_src_to_dst_row = torch.randint(low=0, high=num_rows * top_k, size=(num_rows * top_k,), device=device,dtype=torch.int32)
    >>>drop_pad_mode = 0
    >>> 
    >>> output = torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2_optional, bias, scales,expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode)
    >>> 
    >>> output.shape
    torch.Size([50, 10])
    >>> output.dtype
    torch.float32
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
        
        def forward(self, expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode):
            return torch_npu.npu_moe_finalize_routing(expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode)

    def main():
        expert_num = 16
        token_len = 10
        top_k = 4
        num_rows = 50
        device =torch.device('npu')
        dtype = torch.float32

        expanded_permuted_rows = torch.randn((num_rows * top_k, token_len), device=device, dtype=dtype)
        skip1 = torch.randn((num_rows, token_len), device=device, dtype=dtype)
        skip2_optional = torch.randn((num_rows, token_len), device=device, dtype=dtype)
        bias = torch.randn((expert_num, token_len), device=device, dtype=dtype)
        scales = torch.randn((num_rows, top_k), device=device, dtype=dtype)
        expert_for_source_row = torch.randint(low=0, high=expert_num, size=(num_rows, top_k), device=device, dtype=torch.int32)
        expanded_src_to_dst_row = torch.randint(low=0, high=num_rows * top_k, size=(num_rows * top_k,), device=device, dtype=torch.int32)
        drop_pad_mode = 0

        model = GMMModel().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False)

        custom_output = model(expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row, drop_pad_mode)
        print(custom_output.shape, custom_output.dtype)

    if __name__ == '__main__':
        main()
    
    # 执行上述代码的输出类似如下
    torch.Size([50, 10]) torch.float32
    ```

