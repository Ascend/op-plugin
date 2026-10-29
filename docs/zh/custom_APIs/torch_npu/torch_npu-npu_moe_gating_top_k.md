# torch_npu.npu_moe_gating_top_k

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- API功能：MoE计算中，对输入x做Sigmoid、Softmax或者SqrtSoftplus计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。支持两种模式：
  - **TopK模式**：对normValue进行TopK排序选择专家。
  - **Hash模式**：根据`input_ids`从`tid2eid`映射表中获取预计算的专家索引，跳过排序步骤直接输出。
- 计算公式：

  **TopK模式：**

    **Step 1: 归一化**

    根据`norm_type`对输入`x`做归一化：

    $$
    normOut =
    \begin{cases}
        \text{SoftMax}(x),      & norm\_type = 0 \\
        \text{Sigmoid}(x),      & norm\_type = 1\ or\ 3 \\
        \sqrt{\text{Softplus}(x)},     & norm\_type = 2
    \end{cases}
    $$

    **Step 2: 加偏置**

    若`bias`不为空，加偏置得到用于选择的值：

    $$
    normValue = normOut + bias
    $$

    否则 $normValue = normOut$。

    **Step 3: 分组筛选**（仅`group_count > 1`时执行）

    将normValue按`group_count`分组，根据`group_select_mode`计算每组得分：

    $$
    groupedValue = Reshape(normValue,\ [batch,\ group\_count,\ -1])
    $$

    $$
    groupScore = \begin{cases}
        ReduceMax(groupedValue,\ dim=-1), & group\_select\_mode = 0 \\
        ReduceSum(TopK(groupedValue,\ k=2,\ dim=-1),\ dim=-1), & group\_select\_mode = 1
    \end{cases}
    $$

    选取得分最高的`k_group`个组，将未选中组的对应位置置为 $-\infty$：

    $$
    groupIdx = TopK(groupScore,\ k=k\_group).indices
    $$

    $$
    normValue = Mask(groupedValue,\ groupIdx,\ fillValue=-\infty)
    $$

    **Step 4: Top-K专家选择**

    对normValue取Top-K得到专家索引，这里只需要expertIdxOut：

    $$
    y, expertIdxOut = TopK(normValue[groupIdx, :],\ k=k)
    $$

    **Step 5: Renorm与缩放**

    根据expertIdxOut从normOut中取出对应的k个专家得分：

    $$
    gathered = normOut[\text{expertIdxOut}]
    $$

    `norm_type`为`1`或`2`时始终做归一化；`norm_type`为`0`或`3`时由`renorm`控制是否归一化：

    $$
    if\ (norm\_type = 1\ or\ norm\_type = 2)\ or\ ((norm\_type = 0\ or\ norm\_type = 3)\ and\ renorm = 1):
    $$

    $$
    \quad yOut = \frac{gathered}{ReduceSum(gathered,\ dim=-1) + eps}
    $$

    否则 $yOut = gathered$。

    最终输出：

    $$
    yOut = yOut \times routed\_scaling\_factor
    $$

    **Step 6: 可选输出**

    若`out_flag`为`True`，第三个输出为normOut；否则为空。

  **Hash模式：**

    当同时提供`input_ids`和`tid2eid`时，启用Hash模式。

    **Step 1: 归一化**

    根据`norm_type`对输入`x`做归一化（与TopK模式相同）：

    $$
    normOut =
    \begin{cases}
        \text{SoftMax}(x),      & norm\_type = 0 \\
        \text{Sigmoid}(x),      & norm\_type = 1\ or\ 3 \\
        \sqrt{\text{Softplus}(x)},     & norm\_type = 2
    \end{cases}
    $$

    **Step 2: Hash索引查找**

    根据`input_ids`从`tid2eid`映射表获取专家索引：

    $$
    expertIdxOut = tid2eid[input\_ids, :]
    $$

    其中`tid2eid`的shape为`[numKeys, k]`，`input_ids`的shape为`[batch]`（与`x`的第一维相等），`input_ids`中每个值对应`tid2eid`的一行k个专家索引。

    **Step 3: Gather与缩放**

    根据expertIdxOut从normOut中取出对应的k个专家得分：

    $$
    gathered = normOut[\text{expertIdxOut}]
    $$

    `norm_type`为`1`或`2`时始终做归一化；`norm_type`为`0`或`3`时由`renorm`控制是否归一化：

    $$
    if\ (norm\_type = 1\ or\ norm\_type = 2)\ or\ ((norm\_type = 0\ or\ norm\_type = 3)\ and\ renorm = 1):
    $$

    $$
    \quad yOut = \frac{gathered}{ReduceSum(gathered,\ dim=-1) + eps}
    $$

    否则 $yOut = gathered$。

    最终输出：

    $$
    yOut = yOut \times routed\_scaling\_factor
    $$

- 等价计算逻辑：

    ```python
    import torch
    import numpy

    def moe_gating_top_k_numpy(x: torch.tensor, k: int, *, bias: torch.tensor = None, input_ids: torch.tensor = None,
                                tid2eid: torch.tensor = None, k_group: int = 1, group_count: int = 1,
                                group_select_mode: int = 0, renorm: int = 0, norm_type: int = 0, out_flag: bool = False,
                                routed_scaling_factor: float = 1.0, eps: float = 1e-20) -> tuple:
        dtype = x.dtype
        if dtype != torch.float32:
            x = x.to(dtype=torch.float32)
            if bias is not None:
                bias = bias.to(dtype=torch.float32)

        x = x.numpy()
        bias = bias.numpy() if bias is not None else None
        if norm_type == 0:
            x = x - numpy.max(x, axis=-1, keepdims=True)
            x = numpy.exp(x) / numpy.sum(numpy.exp(x), axis=-1, keepdims=True)  # softmax
        elif norm_type in (1, 3):
            x = 1 / (1 + numpy.exp(-x))  # sigmoid
        else:
            x = numpy.sqrt(numpy.maximum(x, 0) + numpy.log1p(numpy.exp(-numpy.abs(x))))  # sqrt softplus
        original_x = x
        if bias is not None:
            x = x + bias

        hash_flag = input_ids is not None and tid2eid is not None
        if hash_flag:
            # Hash模式：根据input_ids从tid2eid映射表中获取专家索引
            indices = tid2eid.numpy()[input_ids.numpy()].astype(numpy.int64)
        else:
            if group_count > 1:
                x = x.reshape(x.shape[0], group_count, -1)
                if group_select_mode == 0:
                    group_x = numpy.amax(x, axis=-1)
                else:
                    group_x = numpy.partition(x, -2, axis=-1)[..., -2:].sum(axis=-1)
                indices = numpy.argsort(-group_x, axis=-1, kind='stable')[:, :k_group]  # Indices of top-k_group

                mask = numpy.ones((x.shape[0], group_count), dtype=bool)  # Create a mask with all 1
                mask[numpy.arange(x.shape[0])[:, None], indices] = False  # Set to false at the indices
                x = numpy.where(mask[..., None], float('-inf'), x)  # Fill with -inf when mask value is true
                x = x.reshape(x.shape[0], -1)

            indices = numpy.argsort(-x, axis=-1, kind='stable')
            indices = indices[:, :k]
        y = numpy.take_along_axis(original_x, indices, axis=1)

        if norm_type in (1, 2) or (norm_type in (0, 3) and renorm != 0):
            y /= (numpy.sum(y, axis=-1, keepdims=True) + eps)
        y *= routed_scaling_factor
        if out_flag:
            out = original_x
        else:
            out = None

        y = torch.tensor(y, dtype=dtype)
        return y, indices.astype(numpy.int32), out   

    k = 6
    k_group = 4
    group_count = 8
    group_select_mode = 1
    renorm = 0
    norm_type = 1
    out_flag = False
    routed_scaling_factor = 1.0
    eps = 1e-20
        

    x = numpy.random.uniform(-2, 2, (8, 256)).astype(numpy.float32)
    bias = numpy.random.uniform(-2, 2, (256,)).astype(numpy.float32)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    bias_tensor = torch.tensor(bias, dtype=torch.float32)

    y, expert_idx, out = moe_gating_top_k_numpy(x_tensor, k, bias = bias_tensor, k_group = k_group, group_count = group_count,
                                group_select_mode = group_select_mode, renorm = renorm, norm_type = norm_type, out_flag = out_flag,
                                routed_scaling_factor = routed_scaling_factor, eps = eps)

    print(f"yOut shape: {y.shape}")              
    print(f"expertIdxOut shape: {expert_idx.shape}")  
    print(f"Selected experts: {expert_idx[0]}")

    # Hash模式
    k = 2
    norm_type = 1
    input_ids = numpy.random.randint(0, 16, (8,)).astype(numpy.int64)
    tid2eid = numpy.random.randint(0, 256, (16, 2)).astype(numpy.int64)

    y, expert_idx, out = moe_gating_top_k_numpy(x_tensor, k, input_ids = torch.tensor(input_ids),
                                tid2eid = torch.tensor(tid2eid), norm_type = norm_type,
                                routed_scaling_factor = routed_scaling_factor, eps = eps)

    print(f"yOut shape (hash mode): {y.shape}")
    print(f"Selected experts (hash mode): {expert_idx[0]}")
    ```

## 函数原型

```python
npu_moe_gating_top_k(x, k, *, bias=None, input_ids=None, tid2eid=None, k_group=1, group_count=1, group_select_mode=0, renorm=0, norm_type=0, out_flag=False, routed_scaling_factor=1.0, eps=1e-20) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **x**（`Tensor`）：必选参数，表示待计算的输入。要求是一个2D的Tensor，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，数据格式要求为ND。支持非连续Tensor。最后一维的大小（即专家数）要求不大于`2048`。

- **k**（`int`）：必选参数，表示每个token最终筛选得到的专家个数，数据类型为`int64`。要求`1 <= k <= x_shape[-1] / group_count * k_group`。

- <strong>*</strong>：语法分隔符，用于区分位置参数和关键字参数。其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
 
- **bias**（`Tensor`）：可选参数，表示与输入`x`进行计算的bias值。要求是1D的Tensor，要求shape值与`x`的最后一维相等。数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，数据类型需要与`x`保持一致，数据格式要求为ND。支持非连续`Tensor`。

- **input_ids**（`Tensor`）：可选参数，表示Hash模式的输入索引，用于从`tid2eid`中查找专家索引。要求是1D的Tensor，要求shape值与`x`的第一维相等。数据类型支持`torch.int32`、`torch.int64`，数据格式要求为ND。支持非连续`Tensor`。

- **tid2eid**（`Tensor`）：可选参数，表示Hash映射表，存储预计算的专家索引。要求是2D的Tensor，要求shape值的最后一维与`k`相等。数据类型支持`torch.int32`、`torch.int64`，数据格式要求为ND。支持非连续`Tensor`。

- **k_group**（`int`）：可选参数，表示每个token组筛选过程中，选出的专家组个数，数据类型为`int64`，默认值为`1`。要求`1 <= k_group <= group_count`，并且`k_group * x_shape[-1] / group_count`的值要大于等于`k`。

- **group_count**（`int`）：可选参数，表示将全部专家划分的组数，数据类型为`int64`，默认值为`1`。要求group_count > 0，x_shape[-1]能够被`group_count`整除且整除后的结果大于`group_select_mode`，并且整除的结果按照32个数对齐后乘`group_count`的结果不大于`2048`。

- **group_select_mode**（`int`）：可选参数，表示一个专家组的总得分计算方式。默认值为`0`，`0`表示组内取最大值，作为专家组得分；`1`表示取组内Top2的专家进行得分累加，作为专家组得分。

- **renorm**（`int`）：可选参数，表示renorm标记，默认值为`0`。支持`0`和`1`，`0`表示不做renorm，`1`表示做renorm。仅`norm_type`为`0`或`3`时该参数生效；`norm_type`为`1`或`2`时该参数无效，并始终做renorm。
- **norm_type**（`int`）：可选参数，表示norm函数类型，默认值为`0`。

    <!-- npu="A3,910b" id4 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：支持取值`0`、`1`、`2`，`0`表示使用Softmax函数，`1`表示使用Sigmoid函数并做renorm，`2`表示使用SqrtSoftplus函数并做renorm。取值为`2`时要求`group_count`为`1`。
    <!-- end id4 -->
    <!-- npu="950" id5 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持取值`0`、`1`、`2`、`3`，`0`表示使用Softmax函数，`1`表示使用Sigmoid函数并做renorm，`2`表示使用SqrtSoftplus函数并做renorm，`3`表示使用Sigmoid函数且由`renorm`参数控制是否归一化。取值为`2`时对`group_count`无限制。
    <!-- end id5 -->

- **out_flag**（`bool`）：可选参数，是否输出norm函数中间结果。默认值为`False`。
- **routed_scaling_factor**（`float`）：可选参数，表示计算`yOut`使用的`routed_scaling_factor`系数，默认值为`1.0`。
- **eps**（`float`）：可选参数，表示计算`yOut`使用的`eps`系数，默认值为`1e-20`。

## 返回值说明

- **yOut**（`Tensor`）：表示对`x`做norm操作和分组排序topk后计算的结果。要求是一个2D的Tensor，数据类型支持`torch.float16`、`torch.bfloat16`、`torch.float32`，数据类型与`x`需要保持一致，数据格式要求为ND，第一维的大小要求与`x`的第一维相同，最后一维的大小与`k`相同。不支持非连续Tensor。
- **expertIdxOut**（`Tensor`）：表示对`x`做norm操作和分组排序topk后的索引，即专家的序号。shape要求与yOut一致，数据类型支持`torch.int32`，数据格式要求为ND。不支持非连续Tensor。
- **normOut**（`Tensor`）：表示norm计算的输出结果。shape要求与`x`保持一致，数据类型为`torch.float32`，数据格式要求为ND。不支持非连续Tensor。

## 约束说明

- 该接口仅支持推理场景下使用。
- 该接口支持单算子模式和图模式调用。

## 调用示例

- 单算子模式调用

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

- Hash模式单算子调用

    ```python
    import torch
    import torch_npu
    import numpy

    k = 2
    norm_type = 1
    routed_scaling_factor = 1.0
    eps = 1e-20

    # 生成随机数据, 并发送到npu
    x = numpy.random.uniform(0, 2, (16, 256)).astype(numpy.float16)
    x_tensor = torch.tensor(x).npu()
    # input_ids的shape为[batch], 取值范围为[0, numKeys-1]; tid2eid的shape为[numKeys, k]
    input_ids = numpy.random.randint(0, 64, (16,)).astype(numpy.int64)
    tid2eid = numpy.random.randint(0, 256, (64, 2)).astype(numpy.int64)
    input_ids_tensor = torch.tensor(input_ids).npu()
    tid2eid_tensor = torch.tensor(tid2eid).npu()

    # 调用MoeGatingTopK算子（同时传入input_ids和tid2eid时为Hash模式）
    y_npu, expert_idx_npu, out_npu = torch_npu.npu_moe_gating_top_k(x_tensor, k, input_ids=input_ids_tensor, tid2eid=tid2eid_tensor, norm_type=norm_type, routed_scaling_factor=routed_scaling_factor, eps=eps)
    ```

- 图模式调用

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
    