# （beta）torch\_npu.contrib.function.npu\_bbox\_coder\_encode\_xyxy2xywh

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

应用基于NPU的bbox格式编码操作，将格式从xyxy编码为xywh。

## 函数原型

```
torch_npu.contrib.function.npu_bbox_coder_encode_xyxy2xywh(bboxes,gt_bboxes, means=None, stds=None, is_normalized=False, normalized_scale=10000.)
```

## 参数说明

- **bboxes** (`Tensor`)：待转换的框，shape为\(N, 4\)。支持的数据类型为`float`，`half`。
- **gt_bboxes** (`Tensor`)：用作基准的gt\_bboxes，shape为\(N, 4\)。支持的数据类型为`float`，`half`。
- **means** (`List[float]`)：对delta坐标的目标去归一化的方法，默认值为None。
- **stds** (`List[float]`)：对delta坐标的目标去归一化的标准差，默认值为None。
- **is_normalized** (`Bool`)：坐标值是否已归一化，默认值为False。
- **normalized_scale** (`Float`)：设置坐标恢复的归一化比例，默认值为10000.。

## 返回值说明
`Tensor`

代表框转换deltas。

## 约束说明

不支持动态shape。由于算子语义限制，仅支持二维\(n, 4\)场景。`bboxes`和`gt_bboxes`的shape和dtype必须相同，dtype只可为float16和float32。第三个输入（步长）仅支持1D，且第一个维度与第一个输入（`bboxes`）相同。

## 调用示例

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.function import npu_bbox_coder_encode_xyxy2xywh
>>> A = 1024
>>> bboxes = torch.randint(0, 512, size=(A, 4)).npu()
>>> gt_bboxes = torch.randint(0, 512, size=(A, 4)).npu()
>>> out = npu_bbox_coder_encode_xyxy2xywh(bboxes, gt_bboxes)
>>> torch.npu.synchronize()
>>> print('npu_bbox_coder_encode_xyxy2xywh done. output shape is ', out.shape)
npu_bbox_coder_encode_xyxy2xywh done. output shape is torch.Size([1024, 4])
```

