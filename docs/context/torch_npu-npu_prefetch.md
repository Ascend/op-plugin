# torch_npu.npu_prefetch

## 功能说明

提供网络weight预取功能，将需要预取的权重搬到L2 Cache中。尤其在做较大Tensor的MatMul计算且需要搬移到L2 Cache的操作时，可通过该接口提前预取权重，适当提高模型性能，具体效果基于用户对并行的处理。

## 函数原型

```
torch_npu.npu_prefetch(input, dependency, max_size, offset=0) -> ()
```

## 参数说明

- **input** (`Tensor`)：表示需要预取的权重，不做数据处理，与数据类型和数据格式无关；输入不能含有空指针。
- **dependency** (`Tensor`)：表示开始预取的节点，单算子下不生效可为`None`，图模式下不可为`None`；不做数据处理，与数据类型和数据格式无关。
- **max_size** (`int`)：取值需大于0，表示权重预取的最大size，超过预取权重的size时，会设置为权重的最大size。数据类型为`int32`、`int64`。
- **offset** (`int`)：默认值0，取值大于等于0，表示权重预取内存地址偏移，不允许超过权重地址范围。数据类型为`int32`、`int64`。

## 返回值

无

## 约束说明

该接口支持图模式（PyTorch 2.1版本）。

## 支持的型号

<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>

## 调用示例

- 单算子多流并发调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> s_cmo = torch.npu.Stream()
    >>> x = torch.randn(10000, 10000, dtype=torch.float32).npu()
    >>> y = torch.randn(10000, 1, dtype=torch.float32).npu()
    >>> add = torch.add(x, 1)
    >>>
    >>> with torch.npu.stream(s_cmo):
    ...     torch_npu.npu_prefetch(y, None, 10000000)
    ...
    >>> abs = torch.abs(add)
    >>> mul = torch.matmul(abs, abs)
    >>> out = torch.matmul(mul, y)
    >>> out
    [W compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now, dtype cast replace with float. (function operator())
    tensor([[-946066.3750],
            [-945756.1875],
            [-953013.2500],
            ...,
            [-938365.2500],
            [-951188.7500],
            [-941926.4375]], device='npu:0')
    >>> out.shape
    torch.Size([10000, 1])
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)

    x = torch.randn(10000, 10000, dtype=torch.float32).npu()
    y = torch.randn(10000, 1, dtype=torch.float32).npu()


    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            add = torch.add(x, 1)
            torch_npu.npu_prefetch(y, add, 10000000)
            abs = torch.abs(add)
            mul = torch.matmul(abs, abs)
            out = torch.matmul(mul, y)
            return out


    npu_model = Model().npu()
    model = torch.compile(npu_model, backend=npu_backend, dynamic=False, fullgraph=True)
    output = model(x, y)
    print(output)
    print(output.shape)

    # 执行上述代码的输出类似如下    
    tensor([[83962.5078],
            [87820.6328],
            [87498.8594],
            ...,
            [76254.0781],
            [87780.6484],
            [75411.2188]], device='npu:0')
    torch.Size([10000, 1])
    ```

