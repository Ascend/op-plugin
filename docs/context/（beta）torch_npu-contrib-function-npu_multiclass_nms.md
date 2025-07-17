# （beta）torch_npu.contrib.function.npu_multiclass_nms

## 函数原型

```
torch_npu.contrib.function.npu_multiclass_nms(multi_bboxes, multi_scores, score_thr=0.05, nms_thr=0.45, max_num=50, score_factors=None)
```

## 功能说明

使用NPU API的多类bbox NMS。

## 参数说明

- **multi_bboxes** (`Tensor`) - shape(n, \#class, 4)或(n, 4)。
- **multi_scores** (`Tensor`) - shape(n, \#class+1)，其中最后一列包含background class分数，可忽略。在NPU上，为保持语义通顺，我们将统一维度。
- **score_thr** (`Float`，默认值为0.05) - bbox阈值，分数低于它的bbox将不被考虑。
- **nms_thr** (`Float`，默认值为0.45) - NMS IoU阈值。最初的实现是传递\{"iouthreshold": 0.45\}字典，这里做了简化。
- **max_num** (`Int`，默认值为50) - 如果NMS后的bbox数超过max_num值，则只保留最大max_num；如果NMS后的bbox数小于max_num值，则输出将零填充到max_num值。在NPU上需提前申请内存，因此目前不能将max_num值设置为-1。
- **score_factors** (`Tensor`，默认值为None) - NMS应用前用来乘分数的因子。

## 输出说明

**Tuple** - (bboxes, labels)，shape为(k, 5)和(k, 1)的张量。标签以0为基础。

## 约束说明

在动态shape条件下，由于NPU op的限制，最多支持20个类别（nmsed_classes）和10000个框（nmsed_boxes）。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> from torch_npu.contrib.function import npu_multiclass_nms
>>> boxes = torch.randint(1, 255, size=(1000, 4)).npu().half()
>>> scores = torch.randn(1000, 81).npu().half()
>>> det_bboxes, det_labels = npu_multiclass_nms(boxes, scores, score_thr=0.3, nms_thr=0.5, max_num=3)
```

