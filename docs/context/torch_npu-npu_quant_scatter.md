# torch_npu.npu_quant_scatter

## 功能说明

先将`updates`进行量化，然后将`updates`中的值按指定的轴`axis`和索引`indices`更新`input`中的值，并将结果保存到输出tensor，`input`本身的数据不变。

## 函数原型

```
torch_npu.npu_quant_scatter(input, indices, updates, quant_scales, quant_zero_points=None, axis=0, quant_axis=1, reduce='update') -> Tensor
```

## 参数说明

- **input** (`Tensor`)：必选输入，源数据张量，数据类型支持`int8`，数据格式支持$ND$，支持非连续的Tensor，维数只能是3~8维。
- **indices** (`Tensor`)：必选输入，索引张量，数据类型支持`int32`，数据格式支持$ND$，支持非连续的Tensor。
- **updates** (`Tensor`)：必选输入，更新数据张量，数据格式支持$ND$，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float16`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`bfloat16`、`float16`。

- **quant_scales** (`Tensor`)：必选输入，量化缩放张量，数据格式支持$ND$，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float32`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`bfloat16`、`float32`。

- **quant_zero_points** (`Tensor`)：可选输入，量化偏移张量，数据格式支持$ND$，支持非连续的Tensor。
    - <term>Atlas 推理系列产品</term>：数据类型支持`int32`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`bfloat16`、`int32`。

- **axis** (`int`)：可选参数，`updates`上用来更新的轴，默认值为`0`。
- **quant_axis** (`int`)：可选参数，`updates`上用来量化的轴，默认值为`1`。
- **reduce** (`str`)：可选参数，表示数据操作方式；当前只支持`'update'`，即更新操作。

## 返回值
`Tensor`

代表`input`被更新后的结果。

## 约束说明

- 该接口支持图模式（PyTorch 2.1版本）。

- `indices`的维数只能是1维或者2维；如果是2维，其第2维的大小必须是2；不支持索引越界，索引越界不校验；`indices`映射的`input`数据段不能重合，若重合则会因为多核并发原因导致多次执行结果不一样。
- `updates`的维数需要与`input`的维数一样；其第1维的大小等于`indices`的第1维的大小，且不大于`input`的第1维的大小；其`axis`轴的大小不大于`input`的`axis`轴的大小；其余维度的大小要跟`input`对应维度的大小相等；其最后一维的大小必须32B对齐。
- `quant_scales`的元素个数需要等于`updates`在`quant_axis`轴的大小。
- `quant_zero_points`的元素个数需要等于`updates`在`quant_axis`轴的大小。
- `axis`不能为`updates`的第1维或最后1维。
- `quant_axis`只能为`updates`的最后1维。

## 支持的型号

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

- 单算子模式调用

    ```python
   >>> import torch
   >>> import torch_npu
   >>> import numpy as np
   >>>
   >>> data_var = np.random.uniform(0, 1, [24, 4096, 128]).astype(np.int8)
   >>> var = torch.from_numpy(data_var).to(torch.int8).npu()
   >>>
   >>> data_indices = np.random.uniform(0, 1, [24]).astype(np.int32)
   >>> indices = torch.from_numpy(data_indices).to(torch.int32).npu()
   >>>
   >>> data_updates = np.random.uniform(1, 2, [24, 1, 128]).astype(np.float16)
   >>> updates = torch.from_numpy(data_updates).to(torch.bfloat16).npu()
   >>>
   >>> data_quant_scales = np.random.uniform(0, 1, [1, 1, 128]).astype(np.float16)
   >>> quant_scales = torch.from_numpy(data_quant_scales).to(torch.bfloat16).npu()
   >>>
   >>> data_quant_zero_points = np.random.uniform(0, 1, [1, 1, 128]).astype(np.float16)
   >>> quant_zero_points = torch.from_numpy(data_quant_zero_points).to(torch.bfloat16).npu()
   >>>
   >>> axis = -2
   >>> quant_axis = -1
   >>> reduce = "update"
   >>>
   >>> out = torch_npu.npu_quant_scatter(var, indices, updates, quant_scales, quant_zero_points, axis=axis, quant_axis=quant_axis, reduce=reduce)
   >>> out.shape
   torch.Size([24, 4096, 128])
   >>> out.dtype
   torch.int8
   >>> out
   tensor([[[2, 2, 2,  ..., 4, 9, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         [[2, 2, 2,  ..., 5, 6, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         [[3, 2, 3,  ..., 6, 8, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         ...,

         [[2, 2, 2,  ..., 4, 8, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         [[2, 2, 2,  ..., 4, 8, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]],

         [[2, 2, 3,  ..., 4, 9, 2],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            ...,
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0],
            [0, 0, 0,  ..., 0, 0, 0]]], device='npu:0', dtype=torch.int8)
    ```

- 图模式调用

    ```python
    # 入图方式
    import torch
    import torch_npu
    import math
    import torchair as tng
    import numpy as np
    from torchair.configs.compiler_config import CompilerConfig
    import torch._dynamo
    TORCHDYNAMO_VERBOSE=1
    TORCH_LOGS="+dynamo"
    
    # 支持入图的打印宏
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl
    
    # 数据生成
    dtype_list2 =["fp16","int8","int32","fp32","fp16"]
    dtype_list =[np.float16,np.int8,np.int32,np.float32]
    updates_shape =[1,11,1,32]
    var_shape =[1,11,1,32]
    indices_shape =[1,2]
    quant_scales_shape =[1,1,1,32]
    quant_zero_points_shape =[1,1,1,32]
    
    axis =-2
    quant_axis=-1
    reduce = "update"

    updates_data = np.random.uniform(-1,1,size=updates_shape)
    var_data = np.random.uniform(1,2,size=var_shape).astype(dtype_list[1])
    quant_scales_data = np.random.uniform(1,2,size=quant_scales_shape)
    indices_data = np.random.uniform(0,1,size=indices_shape).astype(dtype_list[2])
    quant_zero_points_data = np.random.uniform(0,1,size=quant_zero_points_shape)

    updates_npu = torch.from_numpy(updates_data).npu().to(torch.bfloat16).npu()
    var_npu = torch.from_numpy(var_data).npu()
    quant_scales_npu = torch.from_numpy(quant_scales_data).npu().to(torch.bfloat16).npu()
    quant_zero_points_npu = torch.from_numpy(quant_zero_points_data).to(torch.bfloat16).npu()
    indices_npu = torch.from_numpy(indices_data).npu()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self):
            return torch_npu.npu_quant_scatter(var_npu, indices_npu, updates_npu, quant_scales_npu, quant_zero_points_npu, axis=axis, quant_axis=quant_axis, reduce=reduce)

    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
        single_op = torch_npu.npu_quant_scatter(var_npu, indices_npu, updates_npu, quant_scales_npu, quant_zero_points_npu, axis=axis, quant_axis=quant_axis, reduce=reduce)
        print("single op output with mask:", single_op[0], single_op[0].shape)
        print("graph output with mask:", graph_output[0], graph_output[0].shape)

    if __name__ == "__main__":
        MetaInfershape()
    
    # 执行上述代码的输出类似如下
    single op output with mask: tensor([[[ 1,  1,  0,  1,  0, -1,  0,  0,  0,  1,  0,  1,  0, -1,  1,  0,  0,
               0,  0,  0,  0,  0,  1,  0,  1,  0,  1,  1,  2,  1,  0,  0]],
            [[ 1,  0,  0,  1,  0,  0,  1,  1,  1,  1,  0,  0,  0,  1,  1,  0,  1,
               1,  1,  1,  1,  1,  0,  1,  0,  0,  1,  1,  0,  1,  0,  0]],
            [[ 1,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0, -1,  1,  1,  1,  1,
               0,  1,  0,  2,  0,  0,  0,  1,  0,  1,  1,  2,  0,  1,  1]],
            [[ 1,  1,  0,  1,  0, -1,  0,  1,  0,  1,  0,  0, -1,  0,  1,  0,  0,
               1,  0,  2,  2,  0,  0,  1,  0,  1,  0,  0,  1,  0,  1,  0]],
            [[ 1,  1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  0,  1,  1,  1,  1,  1,
               0,  0,  1,  2,  0,  1,  1,  0,  0,  1,  0,  1,  0,  1,  1]],
            [[ 0,  1,  0,  1,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,  1,  1,  0,
               0,  1,  1,  0, -1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  0]],
            [[ 0,  0,  0,  1,  0,  0,  0,  1,  1,  1,  0,  1,  0, -1,  1,  0,  0,
               1,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  0,  1,  1]],
            [[ 1,  1,  1,  0,  0,  0,  0,  1,  0,  1,  1,  1,  0,  1,  1,  1,  1,
               0,  0,  1,  1,  0,  0,  1,  0,  0,  0,  1,  1,  0,  1,  1]],
            [[ 1,  1,  0,  0,  1,  0,  0,  1,  0,  1,  1,  1,  0,  0,  1, -1,  0,
               1,  1,  0,  0,  1,  0,  1,  1,  0,  0,  1,  0,  1,  1,  1]],
            [[ 1,  0,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  1,  0,  1,
               0,  1,  1,  1, -1,  0,  1,  0,  0,  0,  1,  1,  1,  0,  0]],
            [[ 1,  0, -1,  1,  0,  0,  1,  0,  1,  2,  0,  1,  0, -1,  1,  1,  1,
               1,  0,  0,  2,  1,  0,  1,  1,  0,  1,  0,  1,  0,  1,  0]]],
           device='npu:0', dtype=torch.int8) torch.Size([11, 1, 32])           
    graph output with mask: tensor([[[ 1,  1,  0,  1,  0, -1,  0,  0,  0,  1,  0,  1,  0, -1,  1,  0,  0,
               0,  0,  0,  0,  0,  1,  0,  1,  0,  1,  1,  2,  1,  0,  0]],
            [[ 1,  0,  0,  1,  0,  0,  1,  1,  1,  1,  0,  0,  0,  1,  1,  0,  1,
               1,  1,  1,  1,  1,  0,  1,  0,  0,  1,  1,  0,  1,  0,  0]],
            [[ 1,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0, -1,  1,  1,  1,  1,
               0,  1,  0,  2,  0,  0,  0,  1,  0,  1,  1,  2,  0,  1,  1]],
            [[ 1,  1,  0,  1,  0, -1,  0,  1,  0,  1,  0,  0, -1,  0,  1,  0,  0,
               1,  0,  2,  2,  0,  0,  1,  0,  1,  0,  0,  1,  0,  1,  0]],
            [[ 1,  1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  0,  1,  1,  1,  1,  1,
               0,  0,  1,  2,  0,  1,  1,  0,  0,  1,  0,  1,  0,  1,  1]],
            [[ 0,  1,  0,  1,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,  1,  1,  0,
               0,  1,  1,  0, -1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  0]],
            [[ 0,  0,  0,  1,  0,  0,  0,  1,  1,  1,  0,  1,  0, -1,  1,  0,  0,
               1,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  0,  1,  1]],
            [[ 1,  1,  1,  0,  0,  0,  0,  1,  0,  1,  1,  1,  0,  1,  1,  1,  1,
               0,  0,  1,  1,  0,  0,  1,  0,  0,  0,  1,  1,  0,  1,  1]],
            [[ 1,  1,  0,  0,  1,  0,  0,  1,  0,  1,  1,  1,  0,  0,  1, -1,  0,
               1,  1,  0,  0,  1,  0,  1,  1,  0,  0,  1,  0,  1,  1,  1]],
            [[ 1,  0,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  1,  0,  1,
               0,  1,  1,  1, -1,  0,  1,  0,  0,  0,  1,  1,  1,  0,  0]],
            [[ 1,  0, -1,  1,  0,  0,  1,  0,  1,  2,  0,  1,  0, -1,  1,  1,  1,
               1,  0,  0,  2,  1,  0,  1,  1,  0,  1,  0,  1,  0,  1,  0]]],
           device='npu:0', dtype=torch.int8) torch.Size([11, 1, 32])
    ```

