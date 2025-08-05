# torch_npu.npu_moe_gating_top_k

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> | √   |

## 功能说明

-   API功能：MoE计算中，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。
-   计算公式：

    对输入做sigmoid（`bias`可选）：
    - 当`norm_type`为1时：

      ![](./figures/zh-cn_formulaimage_0000002258672873.png)

    - 当`norm_type`为0时（暂不支持）：

      ![](./figures/zh-cn_formulaimage_0000002313785750.png)

    对计算结果按照groupCount进行分组：
    
    - 如果`group_select_mode`为1，每组按照topk2的sum值对group进行排序，取前kGroup个组：

      ![](./figures/zh-cn_formulaimage_0000002219010398.png)

    - 如果`group_select_mode`为0（暂不支持），每组按照组内的最大值对group进行排序，取前kGroup个组：

      ![](./figures/zh-cn_formulaimage_0000002347834049.png)

    根据上一步的groupId获取`normOut`中对应的元素，将数据再做TopK，得到`expertIdxOut`的结果：

    ![](./figures/zh-cn_formulaimage_0000002219172722.png)

    对y按照输入的`routed_scaling_factor`和`eps`参数进行计算，得到yOut的结果：

    ![](./figures/zh-cn_formulaimage_0000002219173660.png)

## 函数原型

```
npu_moe_gating_top_k(x, k, *, bias=None, k_group=1, group_count=1, group_select_mode=0, renorm=0, norm_type=0, out_flag=False, routed_scaling_factor=1.0, eps=1e-20) -> (Tensor, Tensor, Tensor)
```

## 参数说明

-   **x**（`Tensor`）：必选参数，表示待计算的输入。要求是一个2D的Tensor，数据类型支持`float16`、`bfloat16`、`float32`，数据格式要求为ND。支持非连续Tensor。最后一维的大小（即专家数）当前仅支持取`256`。

-   **k**（`int`）：必选参数，表示每个token最终筛选得到的专家个数，数据类型为`int64`。取值范围为[1, 32]。

-   <strong>*</strong>：代表其之前的变量是位置相关，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
 
-   **bias**（`Tensor`）：可选参数，表示与输入`x`进行计算的bias值。要求是1D的Tensor，要求shape值与`x`的最后一维相等。数据类型支持`float16`、`bfloat16`、`float32`，数据类型需要与`x`保持一致，数据格式要求为ND。支持非连续`Tensor`。必须不为`None`。

-   **k_group**（`int`）：可选参数，表示每个token组筛选过程中，选出的专家组个数，数据类型为`int64`，默认值为`1`。当前仅支持取`4`。

-   **group_count**（`int`）：可选参数，表示将全部专家划分的组数，数据类型为`int64`，默认值为`1`。当前仅支持取`8`。

-   **group_select_mode**（`int`）：可选参数，表示一个专家组的总得分计算方式。默认值为`0`，`0`表示组内取最大值，作为专家组得分；`1`表示取组内Top2的专家进行得分累加，作为专家组得分。当前仅支持`1`。

-   **renorm**（`int`）：可选参数，表示renorm标记，默认值为`0`，表示先进行norm再进行topk计算。当前仅支持`0`。
-   **norm_type**（`int`）：可选参数，表示norm函数类型，`1`表示使用Sigmoid函数，`0`表示Softmax函数。默认值为`0`。当前仅支持`1`。

-   **out_flag**（`bool`）：可选参数，是否输出norm函数中间结果。默认值为`False` ，当前仅支持`False` ，表示不输出。
-   **routed_scaling_factor**（`float`）：可选参数，表示计算`yOut`使用的`routed_scaling_factor`系数，默认值为`1.0`。
-   **eps**（`float`）：可选参数，表示计算`yOut`使用的`eps`系数，默认值为`1e-20`。

## 返回值说明

-   yOut（`Tensor`）：表示对`x`做norm操作和分组排序topk后计算的结果。要求是一个2D的Tensor，数据类型支持`float16`、`bfloat16`、`float32`，数据类型与`x`需要保持一致，数据格式要求为ND，第一维的大小要求与`x`的第一维相同，最后一维的大小与`k`相同。不支持非连续Tensor。
-   expertIdxOut（`Tensor`）：表示对`x`做norm操作和分组排序topk后的索引，即专家的序号。shape要求与yOut一致，数据类型支持`int32`，数据格式要求为ND。不支持非连续Tensor。
-   normOut（`Tensor`）：表示norm计算的输出结果。shape要求与`x`保持一致，数据类型为`float32`，数据格式要求为ND。不支持非连续Tensor。

## 约束说明

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。

## 调用示例

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    import numpy
    
    k = 1
    k_group = 4
    group_count = 8
    group_select_mode = 1
    renorm = 0
    norm_type = 1
    out_flag = False
    routed_scaling_factor = 1.0
    eps = 1e-20
    
    # 生成随机数据, 并发送到npu
    x = numpy.random.uniform(0, 2, (16, 256)).astype(numpy.float16)
    bias = numpy.random.uniform(0, 2, (256,)).astype(numpy.float16)
    x_tensor = torch.tensor(x).npu()
    bias_tensor = torch.tensor(bias).npu()
    
    # 调用MoeGatingTopK算子
    y_npu, expert_idx_npu, out_npu = torch_npu.npu_moe_gating_top_k(x_tensor, k, bias=bias_tensor, k_group=k_group, group_count=group_count, group_select_mode=group_select_mode, renorm=renorm, norm_type=norm_type, out_flag=out_flag, routed_scaling_factor=routed_scaling_factor, eps=eps)
    ```

-   图模式调用

    ```python
    # 入图方式
    import torch
    import torch_npu
    import torchair
    import numpy
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x_tensor, bias_tensor):
            return torch_npu.npu_moe_gating_top_k(x_tensor, k, bias=bias_tensor, k_group=k_group, group_count=group_count, group_select_mode=group_select_mode, renorm=renorm, norm_type=norm_type, out_flag=out_flag, routed_scaling_factor=routed_scaling_factor, eps=eps)
    # 实例化模型model
    model = Model().npu()
    # 从TorchAir获取NPU提供的默认backend
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    # 使用TorchAir的backend去调用compile接口编译模型
    model = torch.compile(model, backend=npu_backend)
    
    k = 1
    k_group = 4
    group_count = 8
    group_select_mode = 1
    renorm = 0
    norm_type = 1
    out_flag = False
    routed_scaling_factor = 1.0
    eps = 1e-20
    
    # 生成随机数据, 并发送到npu
    x = numpy.random.uniform(0, 2, (16, 256)).astype(numpy.float16)
    bias = numpy.random.uniform(0, 2, (256,)).astype(numpy.float16)
    x_tensor = torch.tensor(x).npu()
    bias_tensor = torch.tensor(bias).npu()
    
    # 调用MoeGatingTopK算子
    y_npu, expert_idx_npu, out_npu = model(x_tensor, bias_tensor)
    ```

