# （beta）torch_npu.contrib.function.npu_single_level_responsible_flags

## 函数原型

```
torch_npu.contrib.function.npu_single_level_responsible_flags(featmap_size, gt_bboxes, stride, num_base_anchors)
```

## 功能说明

使用NPU OP在单个特征图中生成锚点的responsible flags。

## 参数说明

- **featmap_size** (`Tuple(Int)`) - 模型总维度。
- **gt_bboxes** (`Tensor`) - 并行attention heads。
- **stride** (`Tuple(Int)`，默认值为None) - key的特性总数。
- **num_base_anchors** (`Int`，默认值为None) - values的特性总数。

## 输出说明

torch.Tensor - 单层特征图中每个锚点的有效标志。输出大小为[featmap_size[0] \* featmap_size[1] \* num_base_anchors]。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> from torch_npu.contrib.function import npu_single_level_responsible_flags
>>> featmap_sizes = [[10, 10], [20, 20], [40, 40]]
>>> stride = [[32, 32], [16, 16], [8, 8]]
>>> gt_bboxes = torch.randint(0, 512, size=(128, 4))
>>> num_base_anchors = 3
>>> featmap_level = len(featmap_sizes)
>>> for i in range(featmap_level):
...     gt_bboxes = gt_bboxes.npu()
>>> out = npu_single_level_responsible_flags(featmap_sizes[i],gt_bboxes,stride[i],num_base_anchors)
>>> print(out.shape, out.max(), out.min())
```

