# （beta）torch_npu.contrib.function.npu_bbox_coder_encode_yolo

## 函数原型

```
torch_npu.contrib.function.npu_bbox_coder_encode_yolo(bboxes, gt_bboxes, stride)
```

## 功能说明

使用NPU OP获取将bbox转换为gt_bbox的框回归转换deltas。

## 参数说明

- bboxes (torch.Tensor) - 源框，例如锚点框。支持数据类型：float、half。
- gt_bboxes (torch.Tensor) - 转换目标框，例如真值框。支持数据类型：float、half。
- stride (torch.Tensor) - bbox步长。仅支持int张量。

## 输出说明

torch.Tensor - 框转换deltas。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> from torch_npu.contrib.function import npu_bbox_coder_encode_yolo
>>> A = 1024
>>> bboxes = torch.randint(0, 512, size=(A, 4)).npu()
>>> gt_bboxes = torch.randint(0, 512, size=(A, 4)).npu()
>>> stride = torch.randint(0, 32, size=(A,)).npu()
>>> out = npu_bbox_coder_encode_yolo(bboxes, gt_bboxes, stride)
>>> torch.npu.synchronize()
>>> print('npu_bbox_coder_encode_yolo done. output shape is ', out.shape)
```

