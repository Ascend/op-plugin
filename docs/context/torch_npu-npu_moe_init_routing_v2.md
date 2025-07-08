# torch\_npu.npu\_moe\_init\_routing\_v2<a name="ZH-CN_TOPIC_0000002309015148"></a>

## 功能说明<a name="zh-cn_topic_0000002271534921_section1650913464367"></a>

-   算子功能：MoE（Mixture of Experts）的routing计算，根据[torch\_npu.npu\_moe\_gating\_top\_k\_softmax](torch_npu-npu_moe_gating_top_k_softmax.md)的计算结果做routing处理，支持不量化和动态量化模式。
-   计算公式：
    1.  对输入expert\_idx做排序，得出排序的结果sorted\_expert\_idx和对应的序号sorted\_row\_idx：

        ![](./figures/zh-cn_formulaimage_0000002320455552.png)

    2.  以sorted\_row\_idx做位置映射得出expanded\_row\_idx：

        ![](./figures/zh-cn_formulaimage_0000002354450121.png)

    3.  在drop模式下，对sorted\_expert\_idx的每个专家统计直方图结果，得出expert\_token\_cumsum\_or\_count：

        ![](./figures/zh-cn_formulaimage_0000002354456273.png)

    4.  动态quant结果：

        若不输入scale：

        ![](./figures/zh-cn_formulaimage_0000002354325361.png)

        若输入scale:

        ![](./figures/zh-cn_formulaimage_0000002320606462.png)

    5.  对quantResult取前NUM\_ROWS个sorted\_row\_idx的对应位置的值，得出输出expanded\_x：

        ![](./figures/zh-cn_formulaimage_0000002320465592.png)

    6.  输出参数expanded\_row\_idx的有效元素数量available\_idx\_num计算方式为，expert\_idx中active\_expert\_range范围内的元素的个数：

        ![](./figures/zh-cn_formulaimage_0000002320606466.png)

## 函数原型<a name="zh-cn_topic_0000002271534921_section14509346133618"></a>

```
torch_npu.npu_moe_init_routing_v2(Tensor x, Tensor expert_idx, *, Tensor? scale=None, Tensor? offset=None, int active_num=-1, int expert_capacity=-1, int expert_num=-1, int drop_pad_mode=0, int expert_tokens_num_type=0, bool expert_tokens_num_flag=False, int quant_mode=0, int[2] active_expert_range=[], int row_idx_type=0) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002271534921_section2050919466367"></a>

-   x：Tensor类型，表示MoE的输入即token特征输入，要求为2D的Tensor，shape为\(NUM\_ROWS, H\)。数据类型支持float16、bfloat16、float32、int8，数据格式要求为ND。
-   expert\_idx：Tensor类型，表示[torch\_npu.npu\_moe\_gating\_top\_k\_softmax](torch_npu-npu_moe_gating_top_k_softmax.md)输出每一行特征对应的K个处理专家，要求是2D的Tensor，shape为\(NUM\_ROWS, K\)，且专家id不能超过专家数。数据类型支持int32，数据格式要求为ND。
-   scale：Tensor类型，可选参数，用于计算量化结果的参数。数据类型支持float32，数据格式要求为ND。如果不输入表示计算时不使用scale，且输出expanded\_scale中的值无意义。
    -   非量化场景下，如果输入则要求为1D的Tensor，shape为\(NUM\_ROWS,\)。
    -   动态quant场景下，如果输入则要求为2D的Tensor，shape为\(expert\_end-expert\_start, H\)。

-   offset：Tensor类型，可选参数，用于计算量化结果的偏移值。数据类型支持float32，数据格式要求为ND。
    -   在非量化场景下不输入。
    -   动态quant场景下不输入。

-   active\_num：int类型，表示总的最大处理row数，输出expanded\_x只有这么多行是有效的，当前入参校验需大于等于0。当前未使用，校验需等于NUM\_ROWS\*K。
-   expert\_capacity：int类型，表示每个专家能够处理的tokens数，取值范围大于等于0。当前未使用，仅校验非空。
-   expert\_num：int类型，表示专家数。expert\_tokens\_num\_type为key\_value模式时，取值范围为\[0, 5120\]；其他模式取值范围为\[0, 10240\]。
-   drop\_pad\_mode：int类型，表示是否为drop\_pad场景。0表示dropless场景，该场景下不校验expert\_capacity。1表示drop\_pad场景。当前仅支持0。
-   expert\_tokens\_num\_type：int类型，取值为0、1和2。0表示cumsum模式。1表示count模式，即输出的值为各个专家处理的token数量的累计值；2表示key\_value模式，即输出的值为专家和对应专家处理token数量的累计值。当前仅支持1和2。
-   expert\_tokens\_num\_flag：bool类型，表示是否输出expert\_token\_cumsum\_or\_count，默认False表示不输出。当前仅支持True。
-   quant\_mode：int类型，表示量化模式，支持取值为0、1、-1。0表示静态量化（默认值，但当前版本暂不支持），-1表示不量化场景；1表示动态quant场景。当前仅支持-1和1。
-   active\_expert\_range：int类型数组，表示活跃expert的范围。数组内值的范围为\[expert\_start, expert\_end\]，表示活跃的expert范围在expert\_start到expert\_end之间。要求值大于等于0，并且expert\_end不大于expert\_num。
-   row\_idx\_type：int类型，表示输出expanded\_row\_idx使用的索引类型，支持取值0和1，默认值0。0表示gather类型的索引；1表示scatter类型的索引。当前仅支持-1和1。

## 输出说明<a name="zh-cn_topic_0000002271534921_section18510124618368"></a>

-   expanded\_x：Tensor类型，根据expert\_idx进行扩展过的特征，要求是2D的Tensor，shape为\(NUM\_ROWS\*K, H\)。非量化场景下数据类型同x；量化场景下数据类型支持int8。数据格式要求为ND。前available\_idx\_num\*H个元素为有效数据，其余为无效数据。其中available\_idx\_num为expert\_idx中active\_expert\_range范围的元素的个数。量化场景下，当x的数据类型为int8时，输出值无意义。
-   expanded\_row\_idx：Tensor类型，expanded\_x和x的映射关系，要求是1D的Tensor，shape为\(NUM\_ROWS\*K, \)，数据类型支持int32，数据格式要求为ND。前available\_idx\_num\*H个元素为有效数据，其余由row\_idx\_type决定。其中available\_idx\_num为expert\_idx中active\_expert\_range范围的元素的个数。量化场景下，当x的数据类型为int8时，输出值无意义。
-   expert\_token\_cumsum\_or\_count：Tensor类型。在expert\_tokens\_num\_type为1的场景下，要求是1D的Tensor，表示active\_expert\_range范围内expert对应的处理token的总数。shape为\(expert\_end-expert\_start, \)；在expert\_tokens\_num\_type为2的场景下，要求是2D的Tensor，shape为\(expert\_num,  2\)，表示active\_expert\_range范围内token总数为非0的expert，以及对应expert处理token的总数；expert id在active\_expert\_range范围且剔除对应expert处理token为0的元素对为有效元素对，存放于Tensor头部并保持原序。数据类型支持int64，数据格式要求为ND。
-   expanded\_scale：Tensor类型，数据类型支持float32，数据格式要求为ND。令available\_idx\_num为active\_expert\_range范围的元素的个数。
    -   非量化场景下，shape为\(NUM\_ROWS\*H\*K,\)。当scale未输入时，输出值无意义；当scale输入时，输出表示一个1D的Tensor，前available\_idx\_num\*H个元素为有效数据，其余为无效数据。
    -   动态quant场景下，输出量化计算过程中scale的中间值，shape为\(NUM\_ROWS \*K\)。当scale未输入时，输出值无意义，输出表示一个1D的Tensor，前available\_idx\_num个元素为有效数据，其余为无效数据，若x的输入类型为int8，输出值无意义。

## 约束说明<a name="zh-cn_topic_0000002271534921_section75102046193618"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。
-   不支持静态量化模式。
-   进入低时延性能模板需要同时满足以下条件：
    -   x、expert\_idx、scale输入Shape要求分别为：\(1, 7168\)、\(1, 8\)、\(256, 7168\)
    -   x数据类型要求：bfloat16
    -   属性要求：active\_expert\_range=\[0, 256\]、 quant\_mode=1、expert\_tokens\_num\_type=2、expert\_num=256

-   进入大batch性能模板需要同时满足以下条件：
    -   NUM\_ROWS范围为\[1920, 4608\]
    -   K=8
    -   expert\_num=256
    -   expert\_end-expert\_start<=32
    -   quant\_mode=-1
    -   row\_idx\_type=1
    -   expert\_tokens\_num\_type=1

## 支持的型号<a name="zh-cn_topic_0000002271534921_section1951034683619"></a>

-   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>

-   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

## 调用示例<a name="zh-cn_topic_0000002271534921_section12510194643618"></a>

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    
    bs = 1
    h = 613
    k = 475
    active_num = 475
    expert_capacity = -1
    expert_num = 226
    drop_pad_mode = 0
    expert_tokens_num_type = 1
    expert_tokens_num_flag = True
    quant_mode = -1
    active_expert_range = [23, 35]
    row_idx_type = 0
    
    x = torch.randn((bs, h), dtype=torch.float32).npu()
    expert_idx = torch.randint(0, expert_num, (bs, k), dtype=torch.int32).npu()
    scale = torch.randn((bs,), dtype=torch.float32).npu()
    offset = None
    
    expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale = torch_npu.npu_moe_init_routing_v2(
                    x, expert_idx, scale=scale, offset=offset,
                    active_num=active_num, expert_capacity=expert_capacity, expert_num=expert_num, drop_pad_mode=drop_pad_mode, 
                    expert_tokens_num_type=expert_tokens_num_type, expert_tokens_num_flag=expert_tokens_num_flag,
                    active_expert_range=active_expert_range, quant_mode=quant_mode, row_idx_type=row_idx_type)
    ```

-   图模式调用

    ```python
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    
    class MoeInitRoutingV2Model(nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, x, expert_idx, *, scale=None, offset=None, active_num=-1, expert_capacity=-1,
                    expert_num=-1, drop_pad_mode=0, expert_tokens_num_type=0, expert_tokens_num_flag=False,
                    quant_mode=0, active_expert_range=0, row_idx_type=0):
            return torch.ops.npu.npu_moe_init_routing_v2(x, expert_idx, scale=scale, offset=offset,
                    active_num=active_num, expert_capacity=expert_capacity, expert_num=expert_num, drop_pad_mode=drop_pad_mode, 
                    expert_tokens_num_type=expert_tokens_num_type, expert_tokens_num_flag=expert_tokens_num_flag,
                    active_expert_range=active_expert_range, quant_mode=quant_mode, row_idx_type=row_idx_type)
    
    def main():
        bs = 1
        h = 613
        k = 475
    
        active_num = 475
        expert_capacity = -1
        expert_num = 226
        drop_pad_mode = 0
        expert_tokens_num_type = 1
        expert_tokens_num_flag = True
        quant_mode = -1
        active_expert_range = [23, 35]
        row_idx_type = 0
    
        x = torch.randn((bs, h), dtype=torch.float32).npu()
        expert_idx = torch.randint(0, expert_num, (bs, k), dtype=torch.int32).npu()
        scale = torch.randn((bs,), dtype=torch.float32).npu()
        offset = None
    
        moe_init_routing_v2_model = MoeInitRoutingV2Model().npu()
        moe_init_routing_v2_model = torch.compile(moe_init_routing_v2_model, backend=npu_backend, dynamic=False)
        expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale = moe_init_routing_v2_model(x,
                                        expert_idx, scale=scale, offset=offset, active_num=active_num,
                                        expert_capacity=expert_capacity, expert_num=expert_num, drop_pad_mode=drop_pad_mode, 
                                        expert_tokens_num_type=expert_tokens_num_type, expert_tokens_num_flag=expert_tokens_num_flag,
                                        active_expert_range=active_expert_range, quant_mode=quant_mode, row_idx_type=row_idx_type)
    
    if __name__ == '__main__':
        main()
    ```

