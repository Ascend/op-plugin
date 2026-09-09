# （beta）torch\_npu.\_npu\_dropout

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 不使用种子（seed）进行dropout结果计数。
- 该接口执行训练过程中的dropout操作：按照概率p随机将输入tensor中的元素置零，并将未被置零的元素按照 1/\(1-p\) 的比例进行缩放，以保持期望值不变。

## 函数原型

```python
torch_npu._npu_dropout(self, p) -> (Tensor, Tensor)
```

## 参数说明

- **self**（`Tensor`）：必选参数，输入张量。数据类型支持`float16`、`float32`、`bfloat16`，Shape支持0-8维。
- **p**（`float`）：必选参数，丢弃概率，取值范围为\[0, 1\]。

## 返回值说明

- **out**（`Tensor`）：dropout后的输出结果，shape和dtype与输入`self`相同。
- **mask**（`Tensor`）：随机生成的mask张量，dtype为`uint8`，用于反向传播。

## 约束说明

- 该接口支持推理、训练场景下使用。
- 该接口支持单算子模式和图模式调用，图模式调用仅适用于<term>Ascend 950PR/Ascend 950DT</term>。

## 调用示例

- 单算子模式调用：

    ```python
    import torch
    import torch_npu
    input = torch.tensor([1.,2.,3.,4.]).npu()
    prob = 0.3
    output, mask = torch_npu._npu_dropout(input, prob)
    ```

- 图模式调用：（仅适用于<term>Ascend 950PR/Ascend 950DT</term>）

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair import patch_for_hcom, get_npu_backend

    # 补丁支持Hcom
    patch_for_hcom()

    # 获取NPU backend
    npu_backend = torchair.get_npu_backend()

    # 定义简单模型
    class Model(torch.nn.Module):
        def forward(self):
            x = torch.randn(4, 8, 16, device='npu')
            m = torch.nn.Dropout(p=0.9)
            return m(x)

    # 实例化模型并移动到NPU
    model = Model().npu()

    # 使用TorchAir编译模型
    opt_model = torch.compile(model, backend=npu_backend)

    # 执行模型
    output = opt_model()
    print("Output shape:", output.shape)
    ```
