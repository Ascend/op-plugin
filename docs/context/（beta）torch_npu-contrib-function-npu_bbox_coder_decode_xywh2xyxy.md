# （beta）torch_npu.contrib.function.npu_bbox_coder_decode_xywh2xyxy

## 函数原型

```
torch_npu.contrib.function.npu_bbox_coder_decode_xywh2xyxy(bboxes, pred_bboxes, means=None, stds=None, max_shape=[9999, 9999], wh_ratio_clip=16 / 1000)
```

## 功能说明

应用基于NPU的bbox格式编码操作，将格式从xywh编码为xyxy。

## 参数说明

- **bboxes** (`torch.Tensor`) - 基础框，shape为(N, 4)。支持dtype：float，half。
- **pred_bboxes** (`torch.Tensor`) - 编码框，shape为(N, 4)。支持dtype：float，half。
- **means** (`List[float]`，默认值为None) - 对delta坐标的目标去归一化的方法。该参数需要与编码参数对齐。
- **stds** (`List[float]`，默认值为None) - 对delta坐标的目标去归一化的标准差。该参数需要与编码参数对齐。
- **max_shape** (`Tuple[int]`，可选，默认为[9999,9999]不受限制)：最大框shape(H, W)，一般对应bbox所在的真实图片的大小。
- **wh_ratio_clip** (`Float`，可选，默认值为16/1000) - 可允许的宽高比。

## 输出说明

Tensor - shape为(N, 4)的框，其中4表示tl_x、tl_y、br_x、br_y。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> from torch_npu.contrib.function import npu_bbox_coder_decode_xywh2xyxy
>>> A = 1024
>>> max_shape = 512
>>> bboxes = torch.randint(0, max_shape, size=(A, 4)).npu()
>>> pred_bboxes = torch.randn(A, 4).npu()
>>> out = npu_bbox_coder_decode_xywh2xyxy(bboxes, pred_bboxes, max_shape=(max_shape, max_shape))
>>> torch.npu.synchronize()
>>> print('npu_bbox_coder_decode_xywh2xyxy done. output shape is ', out.shape)
```

