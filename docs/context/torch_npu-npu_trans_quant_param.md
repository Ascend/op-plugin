# torch_npu.npu_trans_quant_param

## 功能说明

完成量化计算参数`scale`数据类型的转换，将`float32`数据按照bit位存储进一个`int64`数据里。

## 函数原型

```
torch_npu.npu_trans_quant_param(Tensor scale, Tensor? offset=None, int round_mode=0) -> Tensor
```

## 参数说明

- **scale**：`Tensor`类型，数据类型支持`float32`，数据格式支持$ND$，shape支持1维或2维，具体约束参见[约束说明](#zh-cn_topic_0001_section0001)。
- **offset**：`Tensor`类型，可选参数。数据类型支持`float32`，数据格式支持$ND$，shape支持1维或者2维，具体约束参见[约束说明](#zh-cn_topic_0001_section0001)。
- **round_mode**：`int`类型，量化计算中数据类型的转换模式选择，默认为0。0表示截断填充模式（取高19位），1表示R_INT模式（可提升计算精度）。

## 返回值
`Tensor`

代表`trans_quant_param`的计算结果，数据类型支持`int64`。

## 约束说明<a name="zh-cn_topic_0001_section0001"></a>

- 该接口支持推理场景下使用。
- 该接口支持图模式（PyTorch 2.1版本）。
- 该接口在如下产品中支持与matmul类接口（如[torch_npu.npu_quant_matmul](torch_npu-npu_quant_matmul.md)）配套使用。
  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
  - <term>Atlas 推理系列产品</term>
- 当不传入`offset`时，输出shape与`scale` shape一致。
  - 若该输出作为`matmul`类算子输入（如[torch_npu.npu_quant_matmul](torch_npu-npu_quant_matmul.md)），shape支持1维$(1,)$、$(n,)$或2维$(1, n)$，其中$n$与`matmul`计算中右矩阵(`weight`，对应参数x2)的shape $n$一致。
  - 若输出作为`grouped matmul`类算子输入（如[torch_npu.npu_quant_matmul](torch_npu-npu_quant_matmul.md)），仅在分组模式为m轴分组时使用（对应参数`group_type`为0），shape支持1维$(g,)$或2维$(g, 1)$、$(g, n)$，其中$n$与`grouped matmul`计算中右矩阵（对应参数weight）的shape $n$一致，$g$与`grouped matmul`计算中分组数（对应参数`group_list`的shape大小）一致。
- 当传入`offset`时，仅作为`matmul`类算子输入（如[torch_npu.npu_quant_matmul](torch_npu-npu_quant_matmul.md)）:
  - `scale`、`offset`输出的shape支持1维$(1,)$、$(n,)$或2维$(1, n)$，其中$n$与`matmul`计算中右矩阵（`weight`，对应参数x2）的shape $n$一致。
  - 当输入`scale`的shape为1维，输出的shape也为1维，且shape大小为`scale`与`offset`单维shape大小的最大值。
  - 当输入scale的shape为2维，`scale`和`offset`的shape需要保持一致，且输出shape也为$(1, n)$。

## 支持的型号

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> scale = torch.randn(16, dtype=torch.float32)
    >>> offset = torch.randn(16, dtype=torch.float32)
    >>> round_mode = 1
    >>> npu_out = torch_npu.npu_trans_quant_param(scale.npu(), offset.npu(), round_mode)
    >>>
    >>> npu_out
    tensor([ 70507248869376,  70509369614336,  70507209793536, 140463653937152,
            140603250524160, 140603257561088, 140603230814208,  70369813069824,
            70369794605056, 140463675252736,  70784266256384,  70507233009664,
            140601114345472,  70371966238720, 140603258257408, 140603254505472],
        device='npu:0')
    >>> npu_out.dtype
    torch.int64
    >>> npu_out.shape
    torch.Size([16])
    ```~~

- 图模式调用

    图模式下，`npu_trans_quant_param`计算出的结果tensor为`uint64`数据类型。由于torch不支持该数据类型，需要搭配其他接口使用，如示例代码中的`npu_quant_matmul`。

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    import os
    import numpy as np

    os.environ["ENABLE_ACLNN"] = "true"
    config = CompilerConfig()    
    npu_backend = tng.get_npu_backend(compiler_config=config)
    
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, scale, offset, bias, round_mode):
            scale_1 = torch_npu.npu_trans_quant_param(scale, offset, round_mode)
            return torch_npu.npu_quant_matmul(x1, x2, scale_1, offset=offset, bias=bias)

    cpu_model = MyModel()
    model = cpu_model.npu()

    cpu_x1 = torch.randint(-1, 1, (15, 1, 512), dtype=torch.int8)
    cpu_x2 = torch.randint(-1, 1, (15, 512, 128), dtype=torch.int8)
    scale = torch.randn(1, dtype=torch.float32)
    offset = torch.randn(1, dtype=torch.float32)
    round_mode = 1
    bias = torch.randint(-1,1, (15, 1, 128), dtype=torch.int32)
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
    
    npu_out = model(cpu_x1.npu(), cpu_x2.npu(), scale.npu(), offset.npu(), bias.npu(), round_mode)
    print(npu_out.shape)
    print(npu_out)

    # 执行上述代码的输出类似如下
    torch.Size([15, 1, 128])
    tensor([[[62, 56, 58,  ..., 63, 55, 68]],

            [[61, 57, 58,  ..., 60, 50, 53]],

            [[64, 60, 64,  ..., 63, 61, 62]],

            ...,

            [[57, 63, 57,  ..., 63, 61, 62]],

            [[61, 57, 61,  ..., 58, 60, 65]],

            [[68, 62, 63,  ..., 61, 65, 69]]], device='npu:0', dtype=torch.int8)
    ```

