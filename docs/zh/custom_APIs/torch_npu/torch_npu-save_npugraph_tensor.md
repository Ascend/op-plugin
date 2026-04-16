# torch_npu.save_npugraph_tensor

> [!NOTICE]  
> 此接口为本版本新增功能，具体依赖要求请参考《版本说明》中的“[接口变更说明](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/zh/release_notes/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E)”。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √   |
|<term>Atlas 推理系列加速卡产品</term> | √   |

## 功能说明

aclgraph模式下，由于Python原生的print函数包含同步处理，在`torch.compile`时会触发断图，导致图模式下无法使用print观察aclgraph模式执行过程中的tensor值。

当前接口提供了类似原生print特性且不影响aclgraph replay的tensor dump能力，允许将aclgraph中间节点的tensor数据、数据类型、shape信息保存到指定的".pt"或".bin"文件中以便用户观察aclgraph的执行过程，以快速定位问题。

支持单个tensor保存和多个tensor保存（tensorlist/tuple）两种模式。

保存的文件可以配合`torch.load()`接口读取并重建tensor。

## 函数原型

```python
torch_npu.save_npugraph_tensor(input, save_path=None, save_name=None, save_dir=None, suffix=None) -> None
```

## 参数说明

- **input** (`Tensor/List[Tensor]`)：必选参数，用于保存的tensor或tensorlist。当输入为单个tensor时，可以使用此接口保存单个tensor至指定的路径。当输入为tensorlist时，可以将tensorlist中的所有tensor分别保存至指定路径下。
- **save_path** (`str`)：仅支持在需要保存单个tensor时使用，可选参数，文件的完整保存路径。
  - 支持绝对/相对路径，需要具体到文件名，如果路径不存在会优先尝试创建。
  - 支持的文件后缀为".pt"或".bin"。
  - 如果不指定文件保存路径则在当前路径创建，默认命名格式为"tensor_当前时间戳_device_设备序号_计数标识.pt"，例如"**tensor_2026010100000**_device_0_0.pt"。
- **save_name** (`str`)：仅支持在需要保存多个tensor时使用，可选参数，tensors的统一保存文件名主体，用于在保存多个tensors的情况下拼接最终的文件完整保存路径。
  - 被保存的tensorlist中的所有元素会共用文件名主体，并且通过添加序号（index）进行区分。
  - 当用户指定了`save_name`时，命名格式为"保存名称_输入序号_device_设备序号_计数标识"。例如当`save_name`为"inputs"时，最终的文件名主体为"inputs_0_device_0_0"，"inputs_1_device_0_0"等。
  - 如果未指定则默认文件名主体为"tensor_当前时间戳_输入序号_device_设备序号_计数标识"，例如"**tensor_2026010100000_0**_device_0_0"等。
- **save_dir** (`str`)：仅支持在需要保存多个tensor时使用，可选参数，文件的保存路径，用于在保存多个tensors的情况下拼接最终的文件完整保存路径。
  - 支持绝对/相对路径，如果路径不存在会优先尝试创建。
  - 如果未指定则默认为当前路径。
- **suffix** (`str`)：仅支持在需要保存多个tensor时使用，可选参数，文件的保存后缀，用于在保存多个tensors的情况下拼接最终的文件完整保存路径。
  - 可选范围为".pt"或".bin"，如果未指定则默认为".pt"。

## 返回值说明

`None`

无返回值

## 约束说明

- 该接口支持在Eager模式和aclgraph模式下使用。

- 实际的文件路径会加上device序号作为多设备场景下的区分。并且每一个文件拥有一个文件计数标识，作为多个同名文件的区分。例如当多次尝试写入到"tensor.pt"时，会生成多个文件，分别为"tensor_device_x_0.pt"，"tensor_device_x_1.pt"等。

## 调用示例

### 单个tensor dump模式调用示例

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

### 多个tensor dump模式调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>> x = torch.randn([5, 5]).npu()
    >>> y = torch.randn([5, 5]).npu()
    >>> z = torch.randn([5, 5]).npu()
    >>> torch_npu.save_npugraph_tensor.tensorlist((x, y, z), save_name="tensors", save_dir=".", suffix=".pt")
    ```

- 基于`torch.npu.graph`调用

    ```python
    import torch
    import torch_npu

    x = torch.randn([5, 5]).npu()
    y = torch.randn([5, 5]).npu()
    z = torch.randn([5, 5]).npu()

    graph1 = torch.npu.NPUGraph()
    with torch.npu.graph(graph1):
        torch_npu.save_npugraph_tensor.tensorlist((x, y, z), save_name="tensors", save_dir=".", suffix=".pt")
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
        
        def forward(self, x, y, z):
            torch_npu.save_npugraph_tensor((x, y, z), save_name="inputs", save_dir=".", suffix=".pt")
            output = torch.add(x, y)
            output2 = torch.add(output, z)
            torch_npu.save_npugraph_tensor(output2, save_name="outputs", save_dir=".", suffix=".pt")
            return output2

    x = torch.randn([5, 5]).npu()
    y = torch.randn([5, 5]).npu()
    z = torch.randn([5, 5]).npu()
    model = Model()
    model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
    model (x)
    ```

- 保存文件解析：和单tensor dump模式一致。
