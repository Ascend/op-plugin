# （beta）torch\_npu.contrib.function.npu\_bbox\_coder\_encode\_xyxy2xywh

## 函数原型

```
torch_npu.contrib.function.npu_bbox_coder_encode_xyxy2xywh(bboxes, gt_bboxes, means=None, stds=None, is_normalized=False, normalized_scale=10000.)
```

## 功能说明

应用基于NPU的bbox格式编码操作，将格式从xyxy编码为xywh。

## 参数说明

-   **bboxes** (`Tensor`) - 待转换的框，shape为\(N, 4\)。支持dtype：float，half。
-   **gt_bboxes** (`Tensor`) - 用作基准的gt\_bboxes，shape为\(N, 4\)。支持dtype：float，half。
-   **means** (`List[float]`，默认值为None\) - 对delta坐标的目标去归一化的方法。
-   **stds** (`List[float]`，默认值为None\) - 对delta坐标的目标去归一化的标准差。
-   **is_normalized** (`Bool`，默认值为False\) - 坐标值是否已归一化。
-   **normalized_scale** (`Float`，默认值为10000.\) - 设置坐标恢复的归一化比例。

## 输出说明

torch.Tensor - 框转换deltas。

## 约束说明

不支持动态shape。由于算子语义限制，仅支持二维\(n, 4\)场景。bboxes和gt\_bboxes的shape和dtype必须相同，dtype只可为float16和float32。第三个输入（步长）仅支持1D，且第一个维度与第一个输入（bboxes）相同。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>
-   <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> from torch_npu.contrib.function import npu_bbox_coder_encode_xyxy2xywh
>>> A = 1024
>>> bboxes = torch.randint(0, 512, size=(A, 4)).npu()
>>> gt_bboxes = torch.randint(0, 512, size=(A, 4)).npu()
>>> stride = torch.randint(0, 32, size=(A,)).npu()
>>> out = npu_bbox_coder_encode_xyxy2xywh(bboxes, gt_bboxes, stride)
>>> torch.npu.synchronize()
>>> print('npu_bbox_coder_encode_xyxy2xywh done. output shape is ', out.shape)
```

