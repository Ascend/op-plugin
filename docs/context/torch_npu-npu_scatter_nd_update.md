# torch_npu.npu_scatter_nd_update

## 功能说明

将`updates`中的值按指定的索引`indices`更新`input`中的值，并将结果保存到输出tensor，`input`本身的数据不变。

## 函数原型

```
torch_npu.npu_scatter_nd_update(input, indices, updates) -> Tensor
```

## 参数说明

- **input** (`Tensor`)：必选输入，源数据张量，数据格式支持$ND$，支持非连续的Tensor，数据类型需要与`updates`一致，维数只能是1~8维。
    - <term>Atlas 推理系列加速卡产品</term>：数据类型支持`float32`、`float16`、`bool`。
    - <term>Atlas 训练系列产品</term>：数据类型支持`float32`、`float16`、`bool`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`float32`、`float16`、`bool`、`bfloat16`、`int64`、`int8`。
    -  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float32`、`float16`、`bool`、`bfloat16`、`int64`、`int8`。

- **indices** (`Tensor`)：必选输入，索引张量，数据类型支持`int32`、`int64`，数据格式支持$ND$，支持非连续的Tensor，`indices`中的索引数据不支持越界。
- **updates** (`Tensor`)：必选输入，更新数据张量，数据格式支持$ND$，支持非连续的Tensor，数据类型需要与`input`一致。
    - <term>Atlas 推理系列加速卡产品</term>：数据类型支持`float32`、`float16`、`bool`。
    - <term>Atlas 训练系列产品</term>：数据类型支持`float32`、`float16`、`bool`。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`float32`、`float16`、`bool`、`bfloat16`、`int64`、`int8`。
    -  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float32`、`float16`、`bool`、`bfloat16`、`int64`、`int8`。

## 返回值
`Tensor`

代表`input`被更新后的结果。

## 约束说明

- 该接口支持图模式（PyTorch 2.1版本）。
- `indices`至少是2维，其最后1维的大小不能超过`input`的维度大小。
- 假设`indices`最后1维的大小是a，则`updates`的shape等于`indices`除最后1维外的shape加上`input`除前a维外的shape。举例：`input`的shape是$(4, 5, 6)$，`indices`的shape是$(3, 2)$，则`updates`的shape必须是$(3, 6)$。

## 支持的型号

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
- <term>Atlas 训练系列产品</term> 
- <term>Atlas 推理系列产品</term> 
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>> import numpy as np
    >>>
    >>> data_var = np.random.uniform(0, 1, [24, 128]).astype(np.float16)
    >>> var = torch.from_numpy(data_var).to(torch.float16).npu()
    >>>
    >>> data_indices = np.random.uniform(0, 12, [12, 1]).astype(np.int32)
    >>> indices = torch.from_numpy(data_indices).to(torch.int32).npu()
    >>>
    >>> data_updates = np.random.uniform(1, 2, [12, 128]).astype(np.float16)
    >>> updates = torch.from_numpy(data_updates).to(torch.float16).npu()
    >>>
    >>> out = torch_npu.npu_scatter_nd_update(var, indices, updates)
    >>> out
    tensor([[0.6475, 0.3469, 0.2915,  ..., 0.7368, 0.8301, 0.1155],
            [0.5308, 0.7754, 0.5967,  ..., 0.2219, 0.0421, 0.2339],
            [1.7646, 1.1406, 1.5127,  ..., 1.3438, 1.8018, 1.0361],
            ...,
            [0.6396, 0.5396, 0.2939,  ..., 0.9409, 0.5161, 0.1169],
            [0.0737, 0.0457, 0.4727,  ..., 0.5068, 0.8418, 0.6104],
            [0.4180, 0.9102, 0.1122,  ..., 0.0540, 0.4041, 0.3889]],
        device='npu:0', dtype=torch.float16)
    >>> out.shape
    torch.Size([24, 128])
    ```

- 图模式调用

    ```python
    import os
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import torch.nn as nn
    import torch
    import numpy as np
    import numpy
    torch_npu.npu.set_compile_mode(jit_compile=True)
    
    os.environ["ENABLE_ACLNN"] = "false"
    
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
    
        def forward(self, var, indices, update):
            # 调用目标接口
            res = torch_npu.npu_scatter_nd_update(var, indices, update)
            return res
    		
    npu_mode = Network()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
    
    dtype = np.float32
    x = [33 ,5]
    indices = [33,25,1]
    update = [33,25,5]
    
    data_x = np.random.uniform(0, 1, x).astype(dtype)
    data_indices = np.random.uniform(0, 10, indices).astype(dtype)
    data_update = np.random.uniform(0, 1, update).astype(dtype)
    
    tensor_x = torch.from_numpy(data_x).to(torch.float16)
    tensor_indices = torch.from_numpy(data_indices).to(torch.int32)
    tensor_update = torch.from_numpy(data_update).to(torch.float16)
    
    # 传参
    print(npu_mode(tensor_x.npu(), tensor_indices.npu(), tensor_update.npu()))


    # 执行上述代码的输出类似如下   
    tensor([[0.6602, 0.4719, 0.8823, 0.8369, 0.8833],
        [0.7993, 0.2986, 0.0251, 0.8555, 0.7559],
        [0.1278, 0.9434, 0.9409, 0.0586, 0.1710],
        ...,
        [0.9399, 0.8940, 0.5708, 0.7319, 0.1566],
        [0.1333, 0.9614, 0.6128, 0.8457, 0.0269],
        [0.2491, 0.0362, 0.5776, 0.6094, 0.1281],
        [0.2092, 0.7417, 0.8862, 0.1210, 0.8130],
        [0.2910, 0.2468, 0.5488, 0.9761, 0.9785]], device='npu:0',
       dtype=torch.float16)
    ```

