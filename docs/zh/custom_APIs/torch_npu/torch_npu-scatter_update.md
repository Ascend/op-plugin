# torch_npu.scatter_update

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>        |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>        |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>        |    √     |
| <term>Atlas 训练系列产品</term>        |    √     |

## 功能说明

将tensor updates中的值按指定的轴axis和索引indices更新tensor data中的值，并将结果保存到输出tensor，data本身的数据不变。

## 函数原型

```python
torch_npu.scatter_update(data, indices, updates, axis) -> Tensor
```

## 参数说明

- **data** (`Tensor`)：必选参数。代表更新前的原数据，`data`只支持2-8维，且维度大小需要与`updates`一致；支持非连续的tensor；数据格式支持$ND$；不支持空Tensor。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`int8`、`float16`、`float32`、`bfloat16`、`int32`。
    - <term>Atlas A3 训练系列产品</term>：数据类型支持`int8`、`float16`、`float32`、`bfloat16`、`int32`。
    - <term>Atlas 训练系列产品</term>：数据类型支持`int8`、`float16`、`float32`、`int32`。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持`torch.int8`、`torch.uint8`、`torch.float16`、`torch.float32`、`torch.bfloat16`、`torch.int32`。

- **indices** (`Tensor`)：必选参数。代表索引，数据类型支持`int32`、`int64`；目前仅支持一维和二维；支持非连续的tensor；数据格式支持$ND$；不支持空Tensor。仅支持非负索引。indices中的索引数据不支持越界。
- **updates** (`Tensor`)：必选参数。代表更新的数据，`updates`的维度大小需要与`data`一致；支持非连续的tensor；数据格式支持$ND$；不支持空Tensor。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`int8`、`float16`、`float32`、`bfloat16`、`int32`。
    - <term>Atlas A3 训练系列产品</term>：数据类型支持`int8`、`float16`、`float32`、`bfloat16`、`int32`。
    - <term>Atlas 训练系列产品</term>：数据类型支持`int8`、`float16`、`float32`、`int32`。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持`torch.int8`、`torch.uint8`、`torch.float16`、`torch.float32`、`torch.bfloat16`、`torch.int32`。

- **axis** (`int`)：必选参数。代表轴，用来表示scatter的维度，数据类型为`int64`，取值范围为\(-data\_rank, data\_rank\)（data\_rank为data的维度数）且axis不能为0。

## 返回值说明

`Tensor`

计算输出，只支持2-8维，且维度大小需要与`data`一致；支持非连续的tensor；数据格式支持$ND$；不支持空Tensor。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`int8`、`float16`、`float32`、`bfloat16`、`int32`。
- <term>Atlas A3 训练系列产品</term>：数据类型支持`int8`、`float16`、`float32`、`bfloat16`、`int32`。
- <term>Atlas 训练系列产品</term>：数据类型支持`int8`、`float16`、`float32`、`int32`。
- <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持`torch.int8`、`torch.uint8`、`torch.float16`、`torch.float32`、`torch.bfloat16`、`torch.int32`。

## 约束说明

- 该接口支持单算子模式和TorchAir图模式。
- `data`与`updates`的秩一致。
- 不支持索引越界，用户需自行确保索引合法，框架不进行越界检查。
- 当操作为update，且indices有重复时，重复位置的结果不保证。
- updates shape的0轴与indices shape的0轴一致。
- indices为0维时，updates shape的0轴为1。
- updates shape的0轴小于等于data shape的0轴。
- `updates`与`data`的shape，除axis轴和0轴以外，其余轴的shape均相同。
- 当indices shape为二维时，shape的1轴需要等于2。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    data = torch.tensor([[[[1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2]]]], dtype = torch.float32).npu()
    indices = torch.tensor ([1],dtype = torch.int64).npu()
    updates = torch.tensor([[[[3,3,3,3,3,3,3,3]]]], dtype = torch.float32).npu()
    out = torch_npu.scatter_update(data, indices, updates, axis = -2)
    ```

- 图模式调用：仅适用于<term>Ascend 950PR/Ascend 950DT</term>

    ```python
    import os
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import torch.nn as nn
    import torch
    import numpy as np
    import numpy
    torch_npu.npu.set_compile_mode(jit_compile = True)

    os.environ["ENABLE_ACLNN"] = "false"
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, var, indices, update, axis):
            res = torch_npu.scatter_update(var, indices, update, axis)
            return res

    npu_mode = Network()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config = config)
    npu_mode = torch.compile(npu_mode, fullgraph = True, backend = npu_backend, dynamic=False)

    dtype = np.float32
    x = [37,58]
    indices = [31]
    update = [31,22]
    axis = 1

    data_x = np.random.uniform(0, 1, x).astype(dtype)
    data_indices = np.random.uniform(0, 10, indices).astype(dtype)
    data_update = np.random.uniform(0, 1, update).astype(dtype)

    tensor_x = torch.from_numpy(data_x).to(torch.float32)
    tensor_indices = torch.from_numpy(data_indices).to(torch.int32)
    tensor_update = torch.from_numpy(data_update).to(torch.float32)

    # 传参
    print(npu_mode(tensor_x.npu(), tensor_indices.npu(), tensor_update.npu(), axis))
    ```
