# （beta）torch_npu.contrib.function.npu_bbox_coder_encode_yolo
## 产品支持情况


| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |
## 功能说明

使用NPU OP获取将bbox转换为gt_bbox的框回归转换deltas。

## 函数原型

```
torch_npu.contrib.function.npu_bbox_coder_encode_yolo(bboxes, gt_bboxes, stride)
```

## 参数说明

- **bboxes** (`Tensor`)：源框，例如锚点框。支持的数据类型为`float`、`half`。
- **gt_bboxes** (`Tensor`)：转换目标框，例如真值框。支持的数据类型为`float`、`half`。
- **stride** (`Tensor`)：bbox步长。仅支持`int`张量。

## 返回值说明
`Tensor`

框转换deltas。

## 调用示例

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.function import npu_bbox_coder_encode_yolo
>>> A = 1024
>>> bboxes = torch.randint(0, 512, size=(A, 4)).npu()
>>> gt_bboxes = torch.randint(0, 512, size=(A, 4)).npu()
>>> stride = torch.randint(0, 32, size=(A,)).npu()
>>> out = npu_bbox_coder_encode_yolo(bboxes, gt_bboxes, stride)
>>> torch.npu.synchronize()
>>> print('npu_bbox_coder_encode_yolo done. output shape is ', out.shape)
npu_bbox_coder_encode_yolo done. output shape is torch.Size([1024, 4])
```

