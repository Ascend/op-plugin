# （beta）torch_npu.npu_anchor_response_flags

## 函数原型

```
torch_npu.npu_anchor_response_flags(self, featmap_size, stride, num_base_anchors) -> Tensor
```

## 功能说明

在单个特征图中生成锚点的责任标志。

## 参数说明

- self (Tensor) - 真值框，shape为[batch, 4]的2D张量。
- featmap_size (ListInt of length 2) - 特征图大小。
- stride (ListInt of length 2) - 当前轴的步长。
- num_base_anchors (Int) - base anchors的数量。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> x = torch.rand(100, 4).npu()
>>> y = torch_npu.npu_anchor_response_flags(x, [60, 60], [2, 2], 9)
>>> y.shape
torch.Size([32400])
```

