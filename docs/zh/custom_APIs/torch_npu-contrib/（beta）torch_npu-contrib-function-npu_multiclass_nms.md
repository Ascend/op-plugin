# （beta）torch_npu.contrib.function.npu_multiclass_nms

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

执行多类别的非极大值抑制（NMS）计算。

## 函数原型

```python
torch_npu.contrib.function.npu_multiclass_nms(multi_bboxes, multi_scores, score_thr=0.05, nms_thr=0.45, max_num=50, score_factors=None)
```

## 参数说明

- **multi_bboxes** (`Tensor`): 必选参数。候选框（bbox）张量，shape为(n, class, 4)或(n, 4)。
- **multi_scores** (`Tensor`): 必选参数。每个候选框的类别得分，shape为(n, class+1)，其中最后一列包含background class分数，可忽略。
- **score_thr** (`float`): 可选参数，默认值为0.05。候选框分数阈值，分数低于它的候选框将不被考虑。
- **nms_thr** (`float`): 可选参数，默认值为0.45。NMS IoU阈值，取值范围为0到1；两个候选框的IoU大于该阈值时，保留分数较高的候选框。
- **max_num** (`int`): 可选参数，默认值为50。如果NMS后的bbox数超过max_num值，则只保留max_num个；如果NMS后的bbox数小于max_num值，则输出将零填充到max_num值，其中无效框的bboxes全为0、labels为0，可通过bboxes全0识别。
- **score_factors** (`Tensor`): 可选参数，默认值为`None`。传入时，使用形状为 `(n,)`的张量，每个值用于缩放对应候选框的所有前景类别得分，与`multi_scores`的第0维逐行对应。

## 返回值说明

`Tuple`

表示候选框和标签（bboxes, labels），其中bboxes的shape为(k, 5)，labels的shape为(k)。标签以0为基础。

## 约束说明

在动态shape条件下，由于NPU算子的限制，最多支持20个类别（multi_scores的class维度不超过20）和10000个框（multi_bboxes的n维不超过10000）。

## 调用示例

示例为1000个候选框提供20个前景类别的分数。最后一列为背景类别分数。NMS后最多保留3个结果，输出包括边界框和分数，以及对应的类别标签。

```python
import torch, torch_npu
from torch_npu.contrib.function import npu_multiclass_nms
boxes = torch.randint(1, 255, size=(1000, 4)).npu().half()
scores = torch.randn(1000, 21).npu().half()
det_bboxes, det_labels = npu_multiclass_nms(boxes, scores, score_thr=0.3, nms_thr=0.5, max_num=3)
print(det_bboxes.shape)
print(det_labels.shape)
```

示例输出如下。

```text
torch.Size([3, 5])
torch.Size([3])
```
