# （beta）torch_npu.contrib.function.npu_batched_multiclass_nms
## 产品支持情况


| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

执行批量多类别的非极大值抑制（NMS）计算。

## 函数原型

```
torch_npu.contrib.function.npu_batched_multiclass_nms(multi_bboxes, multi_scores, score_thr=0.05, nms_thr=0.45, max_num=50, score_factors=None)
```



## 参数说明

- **multi_bboxes** (`Tensor`): 必选参数。候选框（bbox）张量，shape为(bs, n, class, 4)或(bs, n, 4)。
- **multi_scores** (`Tensor`): 必选参数。每个候选框的类别得分，shape为(bs, n, class+1)，其中最后一列包含background class分数，可忽略。
- **score_thr** (`float`): 可选参数，默认值为0.05。候选框分数阈值，分数低于它的候选框将不被考虑。
- **nms_thr** (`float`): 可选参数，默认值为0.45。NMS IoU阈值。
- **max_num** (`int`): 可选参数，默认值为50。如果NMS后的bbox数超过max_num值，则只保留max_num个bbox；如果NMS后的bbox数小于max_num值，则输出将零填充到max_num值。
- **score_factors** (`Tensor`): 可选参数，默认值为None。NMS应用前用来乘分数的因子。

## 返回值说明

`Tuple`

表示候选框和标签(bboxes, labels)，shape为(bs, k, 5)和(bs, k)的张量。标签以0为基础。

## 约束说明

在动态shape条件下，最多支持20个类别（nmsed_classes）和10000个框（nmsed_boxes）。


## 调用示例

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.function import npu_batched_multiclass_nms
>>> boxes = torch.randint(1, 255, size=(4, 200, 80, 4)).npu().half()
>>> scores = torch.randn(4, 200, 81).npu().half()
>>> det_bboxes, det_labels = npu_batched_multiclass_nms(boxes, scores, score_thr=0.3, nms_thr=0.5, max_num=3)
>>> det_bboxes.shape
torch.Size([4, 3, 5])
>>> det_labels.shape
torch.Size([4, 3])
```

