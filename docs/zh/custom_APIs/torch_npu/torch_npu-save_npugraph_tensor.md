# torch_npu.save_npugraph_tensor

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √   |
|<term>Atlas 推理系列加速卡产品</term> | √   |

## 功能说明

aclgraph模式下，由于Python原生的print函数包含同步处理，在`torch.compile时`会触发断图，导致图模式下无法使用print观察aclgraph模式执行过程中的tensor值。

当前接口提供了类似原生print特性且不影响aclgraph replay的tensor dump能力，允许将ACLGraph中间节点的tensor数据、数据类型、shape信息保存到指定的pt或bin文件中以便用户观察aclgraph的执行过程，以快速定位问题。

## 函数原型

```
torch_npu.save_npugraph_tensor(tensor, save_path=None) -> None
```

## 参数说明

- **tensor** (`Tensor`)：必选参数，用于保存的tensor。
- **save_path** (`str`)：可选参数，文件保存路径。

## 返回值说明

`None`

无返回值

## 约束说明

- 该接口支持在Eager模式和aclgraph模式下使用。

- **save_path**参数的可选文件后缀为".pt"或".bin"，实际的文件路径会加上device序号作为多设备场景下的区分。

- 如果不指定文件保存路径则在当前路径创建，默认命名格式为"tensor_" + 当前时间戳的".pt"文件，例如"tensor_2026010100000_device_0.pt"。

- 保存的文件可以配合`torch.load()`接口读取并重建tensor。

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>> x = torch.randn([5, 5]).npu()
    >>> torch_npu.save_npugraph_tensor(x, save_path = "./x.bin")
    >>> y = torch.add(x, x)
    >>> torch_npu.save_npugraph_tensor(y)
    ```

- 基于`torch.npu.graph`调用

    ```python
    import torch
    import torch_npu

    x = torch.randn([5, 5]).npu()

    graph1 = torch.npu.NPUGraph()
    with torch.npu.graph(graph1):
        torch_npu.save_npugraph_tensor(x, save_path = "./x.pt")
        output = torch.square(x)
        torch_npu.save_npugraph_tensor(output, save_path = "./output.pt")

    graph1.replay()
    ```

- 基于`torch.compile`调用

    ```python
    import torch
    import torch_npu

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            x = torch.add(x, x)
            torch_npu.save_npugraph_tensor(x, save_path = "/home/dump/output.pt")
            x = torch.add(x, 2)
            torch_npu.save_npugraph_tensor(x)
            return x

    x = torch.randn([5, 5]).npu()
    model = Model()
    model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
    model (x)
    ```

- 保存文件解析

    ```python
    >>> x = torch.load("./x_device_0.pt")
    >>> print(x)
    >>> print(x.shape)
    ```
