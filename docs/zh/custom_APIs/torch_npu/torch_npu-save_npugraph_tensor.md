# torch_npu.save_npugraph_tensor

> [!NOTICE]  
> 此接口为本版本新增功能，具体依赖要求请参考《版本说明》中的“[接口变更说明](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/release_notes/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E)”。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √   |

## 功能说明

aclgraph模式下，由于原生PyTorch的`torch.save`函数包含同步处理，在`torch.compile`时会触发断图，并且无法被aclgraph捕获，导致图模式下无法观察aclgraph模式执行过程中的tensor数据。

当前接口提供了类似原生PyTorch的`torch.save`接口特性且不影响aclgraph捕获、重放的tensor dump能力，允许将aclgraph中间节点的tensor数据、数据类型、shape信息保存到指定二进制文件中以便用户观察aclgraph的执行过程的tensor数据，以快速定位问题。

支持单个tensor保存和多个tensor（tensorlist）保存。

保存的文件可以配合`torch.load()`接口读取二进制文件并重建tensor。

## 函数原型

```python
torch_npu.save_npugraph_tensor(input, save_path=None, overwrite=False) -> None
```

## 参数说明

- **input** (`Tensor/List[Tensor]`)：必选参数，用于保存的tensor或tensorlist。

- **save_path** (`str`)：可选参数，仅支持在需要保存单个tensor时使用，文件的完整保存路径。
  - 支持绝对/相对路径，需要具体到文件名，如果路径不存在会优先尝试创建。
  - 支持的文件后缀与原生`torch.save`一致，支持".pt"，".pth"，".bin"等后缀。
  - 如果不指定文件保存路径则在默认当前路径保存，默认命名格式为"tensor_当前时间戳_device_设备序号_计数标识.pt"，例如"tensor_2026010100000_device_0_0.pt"。
  
- **overwrite** (`bool`)：可选参数，是否覆盖已有同名文件，默认值为`False`。
  - 当`overwrite=False`时，遇到同名文件自动添加计数标识，例如"tensor_device_0_0.pt"、"tensor_device_0_1.pt"。
  - 当`overwrite=True`时，会直接覆盖同名文件，文件名不添加计数标识，例如"tensor_device_0.pt"。

## 返回值说明

`None`

无返回值。

## 约束说明

- 该接口支持在Eager模式和aclgraph模式下使用。

- 实际的文件路径会加上device序号作为多设备场景下的区分。当`overwrite=False`时，每一个文件拥有一个文件计数标识，作为多个同名文件的区分。例如当多次尝试写入到"tensor.pt"时，会生成多个文件，分别为"tensor_device_0_0.pt"，"tensor_device_0_1.pt"等；当`overwrite=True`时，会直接覆盖为"tensor_device_0.pt"。

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
        torch_npu.save_npugraph_tensor(x, save_path="./x.pt")
        output = torch.square(x)
        torch_npu.save_npugraph_tensor(output, save_path="./output.pt")

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
            torch_npu.save_npugraph_tensor(x, save_path="/home/dump/output.pt")
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
