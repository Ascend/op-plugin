# torch_npu.npu_moe_init_routing_quant

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

-   API功能：MoE（Mixture of Experts）的Routing计算，根据[torch_npu.npu_moe_gating_top_k_softmax](torch_npu-npu_moe_gating_top_k_softmax.md)的计算结果做Routing处理，支持静态量化和动态量化。

- 计算公式：

  1.对输入`expert_idx`做排序，得出排序后的结果`sorted_expert_idx`和对应的序号`sorted_row_idx`：

   $$
   \text{sorted\_expert\_idx}, \text{sorted\_row\_idx} = keyValueSort(\text{expert\_idx})
   $$

  2.以`sorted_row_idx`做位置映射得出`expanded_row_idx`：

   $$
   \text{expanded\_row\_idx}[\text{sorted\_row\_idx}[i]]=i
   $$

  3.在Dropless模式下，对`sorted_expert_idx`的每个专家统计直方图结果，再进行Cumsum，得出`expert_token_cumsum_or_count`：

   $$
   \text{expert\_token\_cumsum\_or\_count}[i]=Cumsum(Histogram(\text{sorted\_expert\_idx}))
   $$

  4.在Drop模式下，对`sorted_expert_idx`的每个专家统计直方图结果，得出`expert_tokens_before_capacity`：

   $$
   \text{expert\_tokens\_before\_capacity}[i]=Histogram(\text{sorted\_expert\_idx})
   $$

  5.计算量化结果：
    - 静态量化：
        $$
        \text{quant\_result} = round((x \cdot scale) + offset)
        $$
    - 动态量化：
        - 若不输入`scale`：
            $$
            \text{\text{expanded\_scale}} = \frac{RowMax(|x|)}{127}
            $$
            $$
            \text{quant\_result} = round(\frac{x}{\text{expanded\_scale}})
            $$
        - 若输入`scale`:
            $$
            \text{expanded\_scale} = \frac{RowMax(|x \cdot scale|)}{127}
            $$
            $$
            \text{quant\_result} = round(\frac{x}{\text{expanded\_scale}})
            $$
  6.对`quant_result`取前NUM\_ROWS个`sorted_row_idx`的对应位置的值，得出`expanded_x`：

   $$
   \text{expanded\_x}[i]=\text{quant\_result}[\text{sorted\_row\_idx}[i]\%NUM\_ROWS]
   $$

## 函数原型

```
torch_npu.npu_moe_init_routing_quant(Tensor x, Tensor expert_idx, *, Tensor? scale=None, Tensor? offset=None, int active_num=1024, int expert_capacity=0, int expert_num=256, int drop_pad_mode=0, int expert_tokens_num_mode=0, bool expert_tokens_before_capacity_flag=False, int quant_mode=1) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

-   **x** (`Tensor`)：表示MoE的输入（即Token特征），要求为二维Tensor，shape为(NUM_ROWS, H)，其中H表示每个Token的特征维度。支持的数据类型为float16、bfloat16和float32，数据格式要求为ND。
-   **expert_idx** (`Tensor`)：表示[torch_npu.npu_moe_gating_top_k_softmax](torch_npu-npu_moe_gating_top_k_softmax.md)的输出中每一行特征所对应的K个处理专家。要求为二维Tensor，shape为(NUM_ROWS, K)，数据类型为int32，数据格式为ND，且支持非连续内存布局。在Drop/Pad场景下，或非Drop/Pad场景但需输出`expert_token_cumsum_or_count`时，其值域范围须为[0, expert_num - 1]；在其他场景下，值须大于等于0。
-   **scale** (`Tensor`)：可选参数，表示用于计算量化结果的缩放系数。在静态量化场景下必须提供，此时应为一维Tensor，shape为(1,)；在动态量化场景下可选，若提供则须为二维Tensor，shape为(expert_num, H)或(1, H)。数据类型支持float32，数据格式要求为ND。
-   **offset** (`Tensor`)：可选参数，表示用于计算量化结果的偏移值。在静态量化场景下必须提供，此时应为一维Tensor，shape为(1,)。数据类型支持float32，数据格式要求为ND。
-   **active_num** (`int`)：表示是否为Active场景。当`drop_pad_mode`为0时该参数生效，取值范围为大于等于0：0表示Dropless场景，大于0表示Active场景，此时用于约束所有专家共同处理的token总量。
-   **expert_capacity** (`int`)：表示每个专家能够处理的token数量，取值范围大于等于0。在Drop/Pad场景下，取值范围为(0, NUM_ROWS)，此时超出该容量的token将被丢弃，不足容量的部分则填充全0 token；在其他场景下，该属性值无效。
-   **expert_num** (`int`)：表示专家数量，取值范围大于等于0。在Drop/Pad场景下，或当`expert_tokens_num_mode`大于0且需要输出`expert_token_cumsum_or_count`时，`expert_num`必须大于0。
-   **drop_pad_mode** (`int`)：表示是否为Drop/Pad场景，取值为0或1。
    - 0：表示非Drop/Pad场景，该场景下不校验`expert_capacity`。
    - 1：表示Drop/Pad场景，需校验`expert_num`和`expert_capacity`；对于每个专家，超出`expert_capacity`的token将被丢弃，不足的部分将填充全0 token。
-   **expert_tokens_num_type** (`int`)：取值为0、1和2。0表示cumsum模式；1表示count模式，即输出的值为各个专家处理的token数量的累计值；2表示key_value模式，即输出的值为专家和对应专家处理token数量的累计值。当前仅支持1和2。
-   **expert_tokens_num_flag** (`bool`)：表示是否输出`expert_token_cumsum_or_count`，默认False表示不输出。当前仅支持True。
-   **expert_tokens_num_mode** (`int`)：用于控制`expert_token_cumsum_or_count`的输出模式，取值为0、1或2。
    - 0：不输出`expert_token_cumsum_or_count`。
    - 1：输出各专家处理token数量的累计和。
    - 2：输出各专家处理的token数量。
-   **expert_tokens_before_capacity_flag** (`bool`)：用于控制是否输出`expert_tokens_before_capacity`，取值为false或true。
    - false：不输出`expert_tokens_before_capacity`。
    - true：输出各专家在Drop操作前处理的token数量。
-   **quant_mode** (`int`)：表示量化模式，取值为0或1。
    - 0：静态量化场景。
    - 1：动态量化场景。

## 输出说明

-   **expanded_x** (`Tensor`)：表示根据expert_idx扩展后的特征。在Dropless/Active场景下为二维Tensor：Dropless场景shape为(NUM_ROWS \* K, H)，Active场景shape为(min(activeNum, NUM_ROWS \* K), H)；在Drop/Pad场景下为三维Tensor，shape为(expert_num, expert_capacity, H)。数据类型为int，数据格式为ND。
-   **expanded_row_idx** (`Tensor`)：表示`expanded_x`与输入`x`之间的行索引映射关系，为一维Tensor，shape为(NUM_ROWS \* K,)，数据类型为int32，数据格式为ND。
-   **expert_token_cumsum_or_count** (`Tensor`)：表示各专家处理的token数量的统计结果或累加值，是否输出由`expert_tokens_num_mode`参数控制。该值仅在非Drop/Pad场景下输出，为一维Tensor，shape为(expert_num,)，数据类型为int32，数据格式为ND。
-   **expert_tokens_before_capacity** (`Tensor`)：表示在Drop操作前各专家处理的token数量统计结果，是否输出由`expert_tokens_before_capacity_flag`参数控制。该值仅在Drop/Pad场景下输出，为一维Tensor，shape为(expert_num,)，数据类型支持int32，数据格式为ND。
-   **expanded_scale** (`Tensor`)：动态量化计算过程中的中间输出值，仅在动态量化场景下输出。为一维Tensor，shape等于`expanded_x`的shape去除最后一维后所有维度元素个数的乘积，数据类型为float32，数据格式为ND。

## 约束说明

-   该接口支持推理场景下使用。
-   该接口不支持图模式。
-   支持动态和静态量化模式。

## 调用示例

-   单算子模式调用

   ```python
   import torch
   import torch_npu

   bs = 1
   h = 613
   k = 475
   active_num = 475
   expert_capacity = 0
   expert_num = 226
   drop_pad_mode = 0
   expert_tokens_num_mode = 2
   expert_tokens_before_capacity_flag = True
   quant_mode = 1

   x = torch.randn((bs, h), dtype=torch.float32).npu()
   expert_idx = torch.randint(0, expert_num, (bs, k), dtype=torch.int32).npu()


   expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expert_tokens_before_capacity, expanded_scale = torch_npu.npu_moe_init_routing_quant(
                  x, expert_idx,
                  active_num=active_num, expert_capacity=expert_capacity, expert_num=expert_num, drop_pad_mode=drop_pad_mode, 
                  expert_tokens_num_mode=expert_tokens_num_mode, expert_tokens_before_capacity_flag=expert_tokens_before_capacity_flag, quant_mode=quant_mode)
   ```
