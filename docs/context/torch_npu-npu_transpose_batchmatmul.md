# torch\_npu.npu\_transpose\_batchmatmul<a name="ZH-CN_TOPIC_0000002350565344"></a>

## 产品支持情况<a name="zh-cn_topic_0000002319693140_section8593133131718"></a>

<a name="zh-cn_topic_0000002319693140_table1659316316174"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002319693140_row2059343171716"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002319693140_p125930301711"><a name="zh-cn_topic_0000002319693140_p125930301711"></a><a name="zh-cn_topic_0000002319693140_p125930301711"></a><span id="zh-cn_topic_0000002319693140_ph12593183191719"><a name="zh-cn_topic_0000002319693140_ph12593183191719"></a><a name="zh-cn_topic_0000002319693140_ph12593183191719"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002319693140_p18593639173"><a name="zh-cn_topic_0000002319693140_p18593639173"></a><a name="zh-cn_topic_0000002319693140_p18593639173"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002319693140_row859317311178"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002319693140_p1559312351715"><a name="zh-cn_topic_0000002319693140_p1559312351715"></a><a name="zh-cn_topic_0000002319693140_p1559312351715"></a><span id="zh-cn_topic_0000002319693140_ph6756134210183"><a name="zh-cn_topic_0000002319693140_ph6756134210183"></a><a name="zh-cn_topic_0000002319693140_ph6756134210183"></a><term id="zh-cn_topic_0000002319693140_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000002319693140_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000002319693140_zh-cn_topic_0000001312391781_term11962195213215"></a><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002319693140_p1559313317178"><a name="zh-cn_topic_0000002319693140_p1559313317178"></a><a name="zh-cn_topic_0000002319693140_p1559313317178"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002319693140_row294304412306"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002319693140_p49437440302"><a name="zh-cn_topic_0000002319693140_p49437440302"></a><a name="zh-cn_topic_0000002319693140_p49437440302"></a><span id="zh-cn_topic_0000002319693140_ph19280164145411"><a name="zh-cn_topic_0000002319693140_ph19280164145411"></a><a name="zh-cn_topic_0000002319693140_ph19280164145411"></a><term id="zh-cn_topic_0000002319693140_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000002319693140_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000002319693140_zh-cn_topic_0000001312391781_term1253731311225"></a><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002319693140_p8877121915317"><a name="zh-cn_topic_0000002319693140_p8877121915317"></a><a name="zh-cn_topic_0000002319693140_p8877121915317"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0000002319693140_section14441124184110"></a>

-   算子功能：完成张量input与张量weight的矩阵乘计算。仅支持三维的Tensor传入。Tensor支持转置，转置序列根据传入的数列进行变更。perm\_x1代表张量input的转置序列，perm\_x2代表张量weight的转置序列，序列值为0的是batch维度，其余两个维度做矩阵乘法。
-   计算公式：

    T1、T2、Ty分别通过参数perm\_x1、perm\_x2、perm\_y描述转置序列。

    ![](figures/zh-cn_formulaimage_0000002328618340.png)

## 函数原型<a name="zh-cn_topic_0000002319693140_section45077510411"></a>

```
torch_npu.npu_transpose_batchmatmul(Tensor input, Tensor weight, *, Tensor? bias=None, Tensor? scale=None, int[]? perm_x1=[0,1,2], int[]? perm_x2=[0,1,2], int[]? perm_y=[1,0,2], int? batch_split_factor=1) -> Tensor
```

## 参数说明<a name="zh-cn_topic_0000002319693140_section112637109429"></a>

-   **input**（Tensor）：必选参数，表示矩阵乘的第一个矩阵。数据类型支持float16、bfloat16、float32。同时-1轴（末轴）<=65535。数据格式支持ND，shape维度支持3维（B, M, K）或者（M, B, K），B的取值范围为\[1, 65536\)。支持非连续的Tensor。
-   **weight**（Tensor）：必选参数，表示矩阵乘的第二个矩阵。数据类型支持float16、bfloat16、float32。同时-1轴（末轴）<=65535。数据格式支持ND，shape维度支持3维（B, K, N），N的取值范围为\[1, 65536\)。支持非连续的Tensor。weight的Reduce维度需要与input的Reduce维度大小相等。
-   **bias**（Tensor）：可选参数，表示矩阵乘的偏置矩阵，当前版本暂不支持该参数，使用默认值即可。
-   **scale**（Tensor）：可选参数，表示量化输入，数据类型支持int64、uint64。数据格式支持ND，shape维度支持1维\(B \* N\)，B\*N的取值范围为\[1, 65536\)。支持非连续的Tensor。
-   **perm\_x1**（List\[int\]）：可选参数，表示矩阵乘的第一个矩阵的转置序列，size大小为3，数据类型为int64，数据格式支持ND，支持\[0, 1, 2\]、\[1, 0, 2\]。
-   **perm\_x2**（List\[int\]）：可选参数，表示矩阵乘的第二个矩阵的转置序列，size大小为3，数据类型为int64，数据格式支持ND，只支持\[0, 1, 2\]。
-   **perm\_y**（List\[int\]）：可选参数，表示矩阵乘输出矩阵的转置序列，size大小为3，数据类型为int64，数据格式支持ND，只支持\[1, 0, 2\]。
-   **batch\_split\_factor**（int）：可选参数，用于指定矩阵乘输出矩阵中N维的切分大小。数据类型支持int32。取值范围为\[1, N\]且能被N整除，默认值为1。注：当scale有值时，batch\_split\_factor只能为1。

## 返回值说明<a name="zh-cn_topic_0000002319693140_section22231435517"></a>

**y**（Tensor）：表示最终计算结果，数据格式支持ND，shape维度支持3维。

-   当输入scale有值时，数据类型仅为int8类型，shape为\(M, 1, B\*N\)；否则数据类型支持float16、bfloat16、float32。
-   当batch\_split\_factor\>1时，shape大小计算公式为\[batch\_split\_factor, M, B\*N/batch\_split\_factor\]。

## 约束说明<a name="zh-cn_topic_0000002319693140_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持单算子模式和图模式（PyTorch 2.1版本）。
-   输入参数Tensor中shape使用的变量说明：
    -   当perm\_x1为\[1, 0, 2\]时，即input矩阵需要转置时，K\*B的取值范围\[1, 65536\)；当perm\_x1为\[0, 1, 2\]时，K需要小于65536。
    -   K和N需要能被128整除。

## 调用示例<a name="zh-cn_topic_0000002319693140_section14459801435"></a>

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    M, K, N, Batch = 32, 512, 128, 16
    x1 = torch.randn((M, Batch, K), dtype=torch.float16)
    x2 = torch.randn((Batch, K, N), dtype=torch.float16)
    batch_split_factor=1
    output = torch_npu.npu_transpose_batchmatmul(x1.npu(), x2.npu(), bias=None, scale=None, perm_x1=(1,0,2), perm_x2=(0,1,2), perm_y=(1,0,2), batch_split_factor=batch_split_factor)
    ```

-   图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    
    torch.npu.set_compile_mode(jit_compile=True)
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    M, K, N, Batch = 32, 512, 128, 16
    x1 = torch.randn((M, Batch, K), dtype=torch.float16)
    x2 = torch.randn((Batch, K, N), dtype=torch.float16)
    
    class MyModel1(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, x1, x2, perm_x1, perm_y, batch_split_factor=1):
            output = torch_npu.npu_transpose_batchmatmul(x1, x2, perm_x1=perm_x1, perm_y=perm_y, batch_split_factor=batch_split_factor)
            output = output.add(1)
            return output
    
    model = MyModel1().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    output = model(x1.npu(), x2.npu(), (1, 0, 2), (1, 0, 2)).to("cpu")
    ```

