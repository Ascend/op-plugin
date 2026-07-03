# (beta) torch_npu.contrib.function.npu_batched_multiclass_nms

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Performs batched multi-class non-maximum suppression (NMS).

## Prototype

```python
torch_npu.contrib.function.npu_batched_multiclass_nms(multi_bboxes, multi_scores, score_thr=0.05, nms_thr=0.45, max_num=50, score_factors=None)
```

## Parameters

- **`multi_bboxes`** (`Tensor`): Required. Bounding box tensor with shape `(bs, n, class, 4)` or `(bs, n, 4)`.
- **`multi_scores`** (`Tensor`): Required. Class scores for each bounding box. This parameter must be 3D with shape `(bs, n, class+1)`. The last column contains the background class score and can be ignored.
- **`score_thr`** (`float`): Optional. Score threshold for bounding boxes. Bounding boxes with scores lower than this threshold are ignored. The default value is `0.05`.
- **`nms_thr`** (`float`): Optional. IoU threshold for NMS. The default value is `0.45`.
- **`max_num`** (`int`): Optional. Maximum number of bounding boxes after NMS. If the number of bounding boxes after NMS exceeds `max_num`, only the first `max_num` bounding boxes are retained. Otherwise, the output is zero-padded to `max_num`. The default value is `50`.
- **`score_factors`** (`Tensor`): Optional. Factors multiplied with the scores before NMS is applied. The default value is `None`.

## Return Values

`Tuple`

Candidate bounding boxes and labels, represented as `(bboxes, labels)`. The shape can be `(bs, k, 5)` or `(bs, k)`. Labels are zero-based.

## Constraints

In dynamic shape scenarios, a maximum of `20` classes (`nmsed_classes`) and `10000` bounding boxes (`nmsed_boxes`) are supported.

## Example

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.function import npu_batched_multiclass_nms
>>> boxes = torch.randint(1, 255, size=(4, 200, 80, 4)).npu().half()
>>> scores = torch.randn(4, 200, 81).npu().half()
>>> det_bboxes, det_labels = npu_batched_multiclass_nms(boxes, scores, score_thr=0.3, nms_thr=0.5, max_num=3)
>>> print(det_bboxes.shape)
torch.Size([4, 3, 5])
>>> print(det_labels.shape)
torch.Size([4, 3])
```
