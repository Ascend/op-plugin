# （beta）torch_npu.contrib.function.npu_batched_multiclass_nms

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

执行批量多类别的非极大值抑制（NMS）计算。

## 函数原型

```python
torch_npu.contrib.function.npu_batched_multiclass_nms(multi_bboxes, multi_scores, score_thr=0.05, nms_thr=0.45, max_num=50, score_factors=None)
```

## 参数说明

- **multi_bboxes** (`Tensor`): 必选参数。候选框（bbox）张量，shape为(bs, n, class, 4)或(bs, n, 4)。
- **multi_scores** (`Tensor`): 必选参数。每个候选框的类别得分，shape为(bs, n, class+1)，其中最后一列包含background class分数，可忽略。
- **score_thr** (`float`): 可选参数，默认值为0.05。候选框分数阈值，与`multi_scores`使用同一尺度，低于阈值的候选框不参与抑制。
- **nms_thr** (`float`): 可选参数，IoU阈值，默认值为0.45。抑制与同类高分框的IoU超过此阈值的候选框。
- **max_num** (`int`): 可选参数，默认值为50。如果NMS后的bbox数超过max_num值，则只保留max_num个bbox；如果NMS后的bbox数小于max_num值，则输出将零填充到max_num值。
- **score_factors** (`Tensor`): 可选参数，默认值为None。NMS应用前用来乘分数的因子。

## 返回值说明

`Tuple`

表示候选框和标签(bboxes, labels)，shape为(bs, k, 5)和(bs, k)的张量。标签以0为基础。

## 约束说明

在动态shape条件下，最多支持20个前景类别（`class`）和10000个候选框（`n`）。

## 调用示例

示例对4张图像批量执行NMS。每张图有200个候选框和16个前景类别，类别分数的最后一列对应背景。每张图最多保留3个结果。

```python
import torch, torch_npu
from torch_npu.contrib.function import npu_batched_multiclass_nms
boxes = torch.randint(1, 255, size=(4, 200, 16, 4)).npu().half()
scores = torch.randn(4, 200, 17).npu().half()
det_bboxes, det_labels = npu_batched_multiclass_nms(boxes, scores, score_thr=0.3, nms_thr=0.5, max_num=3)
print(det_bboxes.shape)
print(det_labels.shape)
```

示例输出如下。

```text
torch.Size([4, 3, 5])
torch.Size([4, 3])
```
