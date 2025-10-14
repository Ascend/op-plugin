# torch_npu.npu_grouped_matmul_finalize_routing

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>    | √  |

## 功能说明<a name="zh-cn_topic_0000002259406069_section14441124184110"></a>

GroupedMatMul和MoeFinalizeRouting的融合算子，GroupedMatMul计算后的输出按照索引做combine动作。

## 函数原型<a name="zh-cn_topic_0000002259406069_section45077510411"></a>

```
torch_npu.npu_grouped_matmul_finalize_routing(x, w, group_list, *, scale=None, bias=None, offset=None, pertoken_scale=None, shared_input=None, logit=None, row_index=None, dtype=None, shared_input_weight=1.0, shared_input_offset=0, output_bs=0, group_list_type=1) -> Tensor
```

## 参数说明<a name="zh-cn_topic_0000002259406069_section112637109429"></a>

- **x** (`Tensor`)：必选参数。矩阵计算的左矩阵，不支持非连续的Tensor。数据类型支持`int8`，数据格式支持$ND$，维度为\(m, k\)。m取值范围为\[1, 16\*1024\*8\]。
- **w** (`Tensor`)：必选参数。矩阵计算的右矩阵，不支持非连续的Tensor。数据类型支持`int8`、`int4`。
    -   A8W8量化场景下，数据格式支持$NZ$，维度为\(e, n1, k1, k0, n0\)，其中k0=16、n0=32，`x` shape中的k和`w` shape中的k1需要满足以下关系：ceilDiv\(k, 16\) = k1，e取值范围\[1, 256\]，k取值为16整倍数，n取值为32整倍数，且n大于等于256。
    -   A8W4量化场景下数据格式支持$ND$，维度为\(e, k, n\)，k支持2048，n只支持7168。

- **group\_list** (`Tensor`)：必选参数。GroupedMatMul的各分组大小。不支持非连续的Tensor。数据类型支持`int64`，数据格式支持$ND$，维度为\(e,\)，e与`w`的e一致。`group_list`的值总和要求≤m。
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **scale** (`Tensor`)：可选参数。矩阵计算反量化参数，对应weight矩阵。A8W8场景下支持per-channel量化方式，不支持非连续的Tensor。数据类型支持`float32`，数据格式支持$ND$，维度\(e, n\)，这里的n=n1\*n0，A8W4量化场景下，数据类型支持`int64`，维度为\(e, 1, n\)。
- **bias** (`Tensor`)：可选参数。矩阵计算的bias参数，不支持非连续的Tensor。数据类型支持`float32`，数据格式支持$ND$，维度为\(e, n\)，只支持A8W4量化场景。
- **offset** (`Tensor`)：可选参数。矩阵计算量化参数的偏移量，不支持非连续的Tensor。数据类型支持`float32`，数据格式支持$ND$，输入维度只支持3维。只支持A8W4量化场景。
- **pertoken\_scale** (`Tensor`)：可选参数。矩阵计算的反量化参数，对应`x`矩阵，per-token量化方式，不支持非连续的Tensor。维度为\(m,\)，m与`x`的m一致。数据类型支持`float32`，数据格式支持$ND$。
- **shared\_input** (`Tensor`)：可选参数。MoE计算中共享专家的输出，需要与MoE专家的输出进行combine操作，不支持非连续的Tensor。数据类型支持`bfloat16`，数据格式支持$ND$，维度\(batch/dp, n\)，n与`scale`的n一致，batch/dp取值范围\[1, 2\*1024\]，batch取值范围\[1, 16\*1024\]。
- **logit** (`Tensor`)：可选参数。MoE专家对各个token的logit大小，矩阵乘的计算输出与该logit做乘法，然后索引进行combine，不支持非连续的Tensor。数据类型支持`float32`，数据格式支持$ND$，维度\(m,\)，m与`x`的m一致。
- **row\_index** (`Tensor`)：可选参数。MoE专家输出按照该row_index进行combine，其中的值即为combine做scatter add的索引，不支持非连续的Tensor。数据类型支持`int32`、`int64`，数据格式支持$ND$，维度为\(m,\)，m与`x`的m一致。
- **dtype** (`ScalarType`)：可选参数。指定GroupedMatMul计算的输出类型。0表示`float32`，1表示`float16`，2表示`bfloat16`。默认值为0。
- **shared\_input\_weight** (`float`)：可选参数。指共享专家与MoE专家进行combine的系数，`shared_input`先与该参数乘，然后再和MoE专家结果累加。默认为1.0。
- **shared\_input\_offset** (`int`)：可选参数。共享专家输出的在总输出中的偏移。默认值为0。
- **output\_bs** (`int`)：可选参数。输出的最高维大小。默认值为0。
- **group\_list\_type** (`List[int]`)：可选参数。GroupedMatMul的分组模式。默认为1，表示count模式；若配置为0，表示cumsum模式，即为前缀和。

## 返回值说明<a name="zh-cn_topic_0000002259406069_section22231435517"></a>

`Tensor`

返回值，不支持非连续的Tensor，输出的数据类型固定为`float32`，维度为\(batch, n\)。

## 约束说明<a name="zh-cn_topic_0000002259406069_section12345537164214"></a>

-   该接口支持推理和训练场景下使用。
-   该接口支持图模式。
-   输入和输出Tensor支持的数据类型组合如下：
    |x|w|group_list|scale|bias|offset|pertoken_scale|shared_input|logit|row_index|y|
    |--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
    |`int8`|`int8`|`int64`|`float32`|None|None|`float32`|`bfloat16`|`float32`|`int64`|`float32`|
    |`int8`|`int8`|`int64`|`float32`|None|None|`float32`|None|None|`int64`|`float32`|
    |`int8`|`int4`|`int64`|`int64`|`float32`|None|`float32`|`bfloat16`|`float32`|`int64`|`float32`|
    |`int8`|`int4`|`int64`|`int64`|`float32`|`float32`|`float32`|`bfloat16`|`float32`|`int64`|`float32`|

## 调用示例<a name="zh-cn_topic_0000002259406069_section14459801435"></a>

-   单算子模式调用

    ```python
    import numpy as np
    import torch
    import torch_npu
    from scipy.special import softmax
     
    m, k, n = 576, 2048, 7168
    batch = 72
    topK = 8
    group_num = 8
     
    x = np.random.randint(-10, 10, (m, k)).astype(np.int8)
    weight = np.random.randint(-10, 10, (group_num, k, n)).astype(np.int8)
    scale = np.random.normal(0, 0.01, (group_num, n)).astype(np.float32)
    pertoken_scale = np.random.normal(0, 0.01, (m, )).astype(np.float32)
    group_list = np.array([batch] * group_num, dtype=np.int64)
    shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
    routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
    logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
    logit = logit.reshape(m)
    row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)
     
    x_clone = torch.from_numpy(x).npu()
    weight_clone = torch.from_numpy(weight).npu()
    weightNz = torch_npu.npu_format_cast(weight_clone, 29)
    scale_clone = torch.from_numpy(scale).npu()
    pertoken_scale_clone = torch.from_numpy(pertoken_scale).npu()
    group_list_clone = torch.from_numpy(group_list).npu()
    shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
    logit_clone = torch.from_numpy(logit).npu()
    row_index_clone = torch.from_numpy(row_index).npu()
    shared_input_offset = batch // 2
    output_bs = batch
    y = torch_npu.npu_grouped_matmul_finalize_routing(x_clone, weightNz,
                group_list_clone, scale=scale_clone, pertoken_scale=pertoken_scale_clone,
                shared_input=shared_input_clone, logit=logit_clone, row_index=row_index_clone,
                shared_input_offset=shared_input_offset, output_bs=output_bs)
    ```

-   图模式调用：

    ```python
    import numpy as np
    import torch
    import torch_npu
    import torchair as tng
    from scipy.special import softmax
    from torchair.configs.compiler_config import CompilerConfig
     
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
     
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, group_list, scale, pertoken_scale, shared_input, logit, row_index, shared_input_offset, output_bs):
            output = torch_npu.npu_grouped_matmul_finalize_routing(x, weight, group_list,
                        scale=scale, pertoken_scale=pertoken_scale, shared_input=shared_input,
                        logit=logit, row_index=row_index, shared_input_offset=shared_input_offset, output_bs=output_bs)
            return output
     
    m, k, n = 576, 2048, 7168
    batch = 72
    topK = 8
    group_num = 8
     
    x = np.random.randint(-10, 10, (m, k)).astype(np.int8)
    weight = np.random.randint(-10, 10, (group_num, k, n)).astype(np.int8)
    scale = np.random.normal(0, 0.01, (group_num, n)).astype(np.float32)
    pertoken_scale = np.random.normal(0, 0.01, (m, )).astype(np.float32)
    group_list = np.array([batch] * group_num, dtype=np.int64)
    shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
    routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
    logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
    logit = logit.reshape(m)
    row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)
     
    x_clone = torch.from_numpy(x).npu()
    weight_clone = torch.from_numpy(weight).npu()
    weightNz = torch_npu.npu_format_cast(weight_clone, 29)
    scale_clone = torch.from_numpy(scale).npu()
    pertoken_scale_clone = torch.from_numpy(pertoken_scale).npu()
    group_list_clone = torch.from_numpy(group_list).npu()
    shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
    logit_clone = torch.from_numpy(logit).npu()
    row_index_clone = torch.from_numpy(row_index).npu()
    shared_input_offset = batch // 2
    output_bs = batch
     
    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x_clone, weightNz, group_list_clone, scale_clone, pertoken_scale_clone, shared_input_clone, logit_clone, row_index_clone, shared_input_offset, output_bs)
    ```

