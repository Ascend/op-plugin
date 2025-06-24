# torch_npu.npu_grouped_matmul_finalize_routing

## 功能说明

GroupedMatMul和MoeFinalizeRouting的融合算子，GroupedMatMul计算后的输出按照索引做combine动作。

## 函数原型

```
torch_npu.npu_grouped_matmul_finalize_routing(Tensor x, Tensor w, Tensor group_list, *, Tensor? scale=None, Tensor? bias=None, Tensor? pertoken_scale=None, Tensor? shared_input=None, Tensor? logit=None, Tensor? row_index=None, ScalarType? dtype=None, float? shared_input_weight=1.0, int shared_input_offset=0, int? output_bs=0, int? group_list_type=1) -> Tensor
```

## 参数说明

- x：一个2D的Device侧Tensor输入，矩阵计算的左矩阵，不支持非连续的Tensor。数据类型支持int8，数据格式支持ND，维度为(m, k)。m取值范围为[1, 16\*1024\*8]，k只支持2048。
- w：一个5D的Device侧Tensor输入，矩阵计算的右矩阵，不支持非连续的Tensor。数据类型支持int8，数据格式支持NZ，维度为(e, n1, k1, k0, n0)，其中k0=16、n0=32， x shape中的k和w shape中的k1需要满足以下关系：ceilDiv(k, 16) = k1，e取值范围[1, 256]。
- group_list：一个1D的Device侧Tensor输入，GroupedMatMul的各分组大小。，不支持非连续的Tensor。数据类型支持int64，数据格式支持ND，维度为(e,)，e与w的e一致。group_list的值总和要求≤m。
- scale：一个2D的Device侧Tensor输入，矩阵计算反量化参数，对应weight矩阵，per-channel量化方式，不支持非连续的Tensor。数据类型支持int32，数据格式支持ND，维度(e, n)，这里的n=n1\*n0，n只支持7168。
- bias：一个2D的Device侧Tensor输入，矩阵计算的bias参数，不支持非连续的Tensor。数据类型支持float32，数据格式支持ND。
- pertoken_scale：一个1D的Device侧Tensor输入，矩阵计算的反量化参数，对应x矩阵，per-token量化方式，不支持非连续的Tensor。维度为(m,)，m与x的m一致。数据类型支持float32，数据格式支持ND。
- shared_input：一个2D的Device侧Tensor输入，MoE计算中共享专家的输出，需要与MoE专家的输出进行combine操作，不支持非连续的Tensor。数据类型支持bfloat16，数据格式支持ND，维度(batch/dp, n)，n与scale的n一致，batch/dp取值范围[1, 2\*1024]，batch取值范围[1, 16\*1024]。
- logit：一个1D的Device侧Tensor输入，MoE专家对各个token的logit大小，矩阵乘的计算输出与该logit做乘法，然后索引进行combine，不支持非连续的Tensor。数据类型支持float32，数据格式支持ND，维度(m,)，m与x的m一致。
- row_index：一个1D的Device侧Tensor输入，MoE专家输出按照该row_index进行combine，其中的值即为combine做scatter add的索引，不支持非连续的Tensor。数据类型支持int64，数据格式支持ND，维度为(m,)，m与x的m一致。
- dtype：ScalarType类型，指定GroupedMatMul计算的输出类型。0表示float32，1表示float16，2表示bfloat16。默认值为0。
- shared_input_weight：float类型，指共享专家与MoE专家进行combine的系数，shared_input先与该参数乘，然后再和MoE专家结果累加。默认为1.0。
- shared_input_offset：int类型，共享专家输出的在总输出中的偏移。默认值为0。
- output_bs：int类型，输出的最高维大小。默认值为0。
- group_list_type：int类型数组，GroupedMatMul的分组模式。默认为1，表示count模式；若配置为0，表示cumsum模式，即为前缀和。

## 输出说明

y：一个2D的Tensor，不支持非连续的Tensor，输出的数据类型固定为float32，维度为(batch, n)。

## 约束说明

- 该接口在推理和训练场景下使用。
- 该接口支持图模式（PyTorch 2.1版本）。
- 输入和输出Tensor支持的数据类型组合如下：

    <a name="zh-cn_topic_0000002259406069_table334073018273"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000002259406069_row13340430162711"><th class="cellrowborder" valign="top" width="7.24%" id="mcps1.1.11.1.1"><p id="zh-cn_topic_0000002259406069_p13340173011275"><a name="zh-cn_topic_0000002259406069_p13340173011275"></a><a name="zh-cn_topic_0000002259406069_p13340173011275"></a>x</p>
    </th>
    <th class="cellrowborder" valign="top" width="6.65%" id="mcps1.1.11.1.2"><p id="zh-cn_topic_0000002259406069_p634110308278"><a name="zh-cn_topic_0000002259406069_p634110308278"></a><a name="zh-cn_topic_0000002259406069_p634110308278"></a>w</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.14%" id="mcps1.1.11.1.3"><p id="zh-cn_topic_0000002259406069_p78611055143112"><a name="zh-cn_topic_0000002259406069_p78611055143112"></a><a name="zh-cn_topic_0000002259406069_p78611055143112"></a>group_list</p>
    </th>
    <th class="cellrowborder" valign="top" width="8.559999999999999%" id="mcps1.1.11.1.4"><p id="zh-cn_topic_0000002259406069_p534163092719"><a name="zh-cn_topic_0000002259406069_p534163092719"></a><a name="zh-cn_topic_0000002259406069_p534163092719"></a>scale</p>
    </th>
    <th class="cellrowborder" valign="top" width="8.58%" id="mcps1.1.11.1.5"><p id="zh-cn_topic_0000002259406069_p734113016272"><a name="zh-cn_topic_0000002259406069_p734113016272"></a><a name="zh-cn_topic_0000002259406069_p734113016272"></a>bias</p>
    </th>
    <th class="cellrowborder" valign="top" width="14.099999999999998%" id="mcps1.1.11.1.6"><p id="zh-cn_topic_0000002259406069_p1534119307276"><a name="zh-cn_topic_0000002259406069_p1534119307276"></a><a name="zh-cn_topic_0000002259406069_p1534119307276"></a>pertoken_scale</p>
    </th>
    <th class="cellrowborder" valign="top" width="12.740000000000002%" id="mcps1.1.11.1.7"><p id="zh-cn_topic_0000002259406069_p12341153019274"><a name="zh-cn_topic_0000002259406069_p12341153019274"></a><a name="zh-cn_topic_0000002259406069_p12341153019274"></a>shared_input</p>
    </th>
    <th class="cellrowborder" valign="top" width="9.76%" id="mcps1.1.11.1.8"><p id="zh-cn_topic_0000002259406069_p1934123012719"><a name="zh-cn_topic_0000002259406069_p1934123012719"></a><a name="zh-cn_topic_0000002259406069_p1934123012719"></a>logit</p>
    </th>
    <th class="cellrowborder" valign="top" width="12.78%" id="mcps1.1.11.1.9"><p id="zh-cn_topic_0000002259406069_p193411530182716"><a name="zh-cn_topic_0000002259406069_p193411530182716"></a><a name="zh-cn_topic_0000002259406069_p193411530182716"></a>row_index</p>
    </th>
    <th class="cellrowborder" valign="top" width="9.45%" id="mcps1.1.11.1.10"><p id="zh-cn_topic_0000002259406069_p4341930152710"><a name="zh-cn_topic_0000002259406069_p4341930152710"></a><a name="zh-cn_topic_0000002259406069_p4341930152710"></a>y</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000002259406069_row19341133042719"><td class="cellrowborder" valign="top" width="7.24%" headers="mcps1.1.11.1.1 "><p id="zh-cn_topic_0000002259406069_p6341113020274"><a name="zh-cn_topic_0000002259406069_p6341113020274"></a><a name="zh-cn_topic_0000002259406069_p6341113020274"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="6.65%" headers="mcps1.1.11.1.2 "><p id="zh-cn_topic_0000002259406069_p6341630172715"><a name="zh-cn_topic_0000002259406069_p6341630172715"></a><a name="zh-cn_topic_0000002259406069_p6341630172715"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.14%" headers="mcps1.1.11.1.3 "><p id="zh-cn_topic_0000002259406069_p78617552312"><a name="zh-cn_topic_0000002259406069_p78617552312"></a><a name="zh-cn_topic_0000002259406069_p78617552312"></a>int64</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.559999999999999%" headers="mcps1.1.11.1.4 "><p id="zh-cn_topic_0000002259406069_p1234163013273"><a name="zh-cn_topic_0000002259406069_p1234163013273"></a><a name="zh-cn_topic_0000002259406069_p1234163013273"></a>float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.58%" headers="mcps1.1.11.1.5 "><p id="zh-cn_topic_0000002259406069_p1341163002713"><a name="zh-cn_topic_0000002259406069_p1341163002713"></a><a name="zh-cn_topic_0000002259406069_p1341163002713"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.099999999999998%" headers="mcps1.1.11.1.6 "><p id="zh-cn_topic_0000002259406069_p133411130172713"><a name="zh-cn_topic_0000002259406069_p133411130172713"></a><a name="zh-cn_topic_0000002259406069_p133411130172713"></a>float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.740000000000002%" headers="mcps1.1.11.1.7 "><p id="zh-cn_topic_0000002259406069_p12341930142712"><a name="zh-cn_topic_0000002259406069_p12341930142712"></a><a name="zh-cn_topic_0000002259406069_p12341930142712"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.76%" headers="mcps1.1.11.1.8 "><p id="zh-cn_topic_0000002259406069_p1341183013277"><a name="zh-cn_topic_0000002259406069_p1341183013277"></a><a name="zh-cn_topic_0000002259406069_p1341183013277"></a>float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.78%" headers="mcps1.1.11.1.9 "><p id="zh-cn_topic_0000002259406069_p434112308271"><a name="zh-cn_topic_0000002259406069_p434112308271"></a><a name="zh-cn_topic_0000002259406069_p434112308271"></a>int64</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.45%" headers="mcps1.1.11.1.10 "><p id="zh-cn_topic_0000002259406069_p113411230182710"><a name="zh-cn_topic_0000002259406069_p113411230182710"></a><a name="zh-cn_topic_0000002259406069_p113411230182710"></a>float32</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002259406069_row10341133020278"><td class="cellrowborder" valign="top" width="7.24%" headers="mcps1.1.11.1.1 "><p id="zh-cn_topic_0000002259406069_p1234143017274"><a name="zh-cn_topic_0000002259406069_p1234143017274"></a><a name="zh-cn_topic_0000002259406069_p1234143017274"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="6.65%" headers="mcps1.1.11.1.2 "><p id="zh-cn_topic_0000002259406069_p434193010273"><a name="zh-cn_topic_0000002259406069_p434193010273"></a><a name="zh-cn_topic_0000002259406069_p434193010273"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.14%" headers="mcps1.1.11.1.3 "><p id="zh-cn_topic_0000002259406069_p148611355113119"><a name="zh-cn_topic_0000002259406069_p148611355113119"></a><a name="zh-cn_topic_0000002259406069_p148611355113119"></a>int64</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.559999999999999%" headers="mcps1.1.11.1.4 "><p id="zh-cn_topic_0000002259406069_p934123015274"><a name="zh-cn_topic_0000002259406069_p934123015274"></a><a name="zh-cn_topic_0000002259406069_p934123015274"></a>float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.58%" headers="mcps1.1.11.1.5 "><p id="zh-cn_topic_0000002259406069_p11341030102719"><a name="zh-cn_topic_0000002259406069_p11341030102719"></a><a name="zh-cn_topic_0000002259406069_p11341030102719"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.099999999999998%" headers="mcps1.1.11.1.6 "><p id="zh-cn_topic_0000002259406069_p1034183032717"><a name="zh-cn_topic_0000002259406069_p1034183032717"></a><a name="zh-cn_topic_0000002259406069_p1034183032717"></a>float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.740000000000002%" headers="mcps1.1.11.1.7 "><p id="zh-cn_topic_0000002259406069_p183411230112713"><a name="zh-cn_topic_0000002259406069_p183411230112713"></a><a name="zh-cn_topic_0000002259406069_p183411230112713"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.76%" headers="mcps1.1.11.1.8 "><p id="zh-cn_topic_0000002259406069_p17341330182711"><a name="zh-cn_topic_0000002259406069_p17341330182711"></a><a name="zh-cn_topic_0000002259406069_p17341330182711"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.78%" headers="mcps1.1.11.1.9 "><p id="zh-cn_topic_0000002259406069_p1634103092711"><a name="zh-cn_topic_0000002259406069_p1634103092711"></a><a name="zh-cn_topic_0000002259406069_p1634103092711"></a>int64</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.45%" headers="mcps1.1.11.1.10 "><p id="zh-cn_topic_0000002259406069_p11342153022712"><a name="zh-cn_topic_0000002259406069_p11342153022712"></a><a name="zh-cn_topic_0000002259406069_p11342153022712"></a>float32</p>
    </td>
    </tr>
    </tbody>
    </table>

## 支持的型号

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> 
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> 

## 调用示例

- 单算子模式调用

    ```python
    import numpy as np
    import torch
    import torch_npu
    import tensorflow as tf
    from scipy.special import softmax
     
    bfloat16 = tf.bfloat16.as_numpy_dtype
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

- 图模式调用：

    ```python
    import numpy as np
    import torch
    import torch_npu
    import torchair as tng
    import tensorflow as tf
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
     
    bfloat16 = tf.bfloat16.as_numpy_dtype
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

