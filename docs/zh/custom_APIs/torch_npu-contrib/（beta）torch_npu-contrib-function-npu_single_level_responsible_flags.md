# （beta）torch_npu.contrib.function.npu_single_level_responsible_flags

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

使用NPU算子在单个特征图中生成锚点的responsible flags（有效标志）。

## 函数原型

```python
torch_npu.contrib.function.npu_single_level_responsible_flags(featmap_size, gt_bboxes, stride, num_base_anchors)
```

## 参数说明

- **featmap_size** (`tuple[int, int] | list[int]`)：特征图的尺寸。
- **gt_bboxes** (`Tensor`)：Ground Truth的边界框。
- **stride** (`tuple[int, int] | list[int]`)：当前等级的步长。
- **num_base_anchors** (`int`)：基础锚点数。

## 返回值说明

`Tensor` 

代表单层特征图中每个锚点的有效标志。输出大小为[featmap_size[0] \* featmap_size[1] \* num_base_anchors]。

> [!NOTE]
>
> 此接口调用的`torch_npu.npu_anchor_response_flags`已标记为废弃，运行时可能提示弃用警告。

## 调用示例

示例对三种分辨率的特征图分别计算锚框匹配标记。图像框作为 `(128, 4)`张量输入，每个位置设置3个基础锚框。

```python
import torch
import torch_npu
from torch_npu.contrib.function import npu_single_level_responsible_flags
featmap_sizes = [[10, 10], [20, 20], [40, 40]]
stride = [[32, 32], [16, 16], [8, 8]]
gt_bboxes = torch.randint(0, 512, size=(128, 4))
num_base_anchors = 3
featmap_level = len(featmap_sizes)
for i in range(featmap_level):
    gt_bboxes = gt_bboxes.npu()
    out = npu_single_level_responsible_flags(featmap_sizes[i],gt_bboxes,stride[i],num_base_anchors)
    print(out.shape, out.max(), out.min())
```

三个输出的长度分别为300、1200和4800，每个元素用`0`或`1`表示锚框是否匹配。
