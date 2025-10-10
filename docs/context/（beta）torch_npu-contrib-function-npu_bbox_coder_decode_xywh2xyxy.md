# （beta）torch_npu.contrib.function.npu_bbox_coder_decode_xywh2xyxy

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

应用基于NPU的bbox格式编码操作，将格式从xywh编码为xyxy。

## 函数原型

```
torch_npu.contrib.function.npu_bbox_coder_decode_xywh2xyxy(bboxes, pred_bboxes, means=None, stds=None, max_shape=[9999, 9999], wh_ratio_clip=16 / 1000)
```

## 参数说明

- **bboxes** (`Tensor`)：基础框，shape为(N, 4)。支持的数据类型为`float`，`half`。
- **pred_bboxes** (`Tensor`)：编码框，shape为(N, 4)。支持的数据类型为`float`，`half`。
- **means** (`List[float]`)：对delta坐标的目标去归一化的方法，默认值为None。该参数需要与编码参数对齐。
- **stds** (`List[float]`)：对delta坐标的目标去归一化的标准差，默认值为None。该参数需要与编码参数对齐。
- **max_shape** (`Tuple[int]`)：可选参数，最大框shape(H, W)，一般对应bbox所在的真实图片的大小，默认为[9999,9999]不受限制。
- **wh_ratio_clip** (`float`)：可选参数，可允许的宽高比，默认值为16/1000。

## 返回值说明

`Tensor`

代表shape为(N, 4)的框，其中4表示tl_x、tl_y、br_x、br_y。

## 调用示例

```python
>>> import troch, torch_npu
>>> from torch_npu.contrib.function import npu_bbox_coder_decode_xywh2xyxy
>>> A = 1024
>>> max_shape = 512
>>> bboxes = torch.randint(0, max_shape, size=(A, 4)).npu()
>>> pred_bboxes = torch.randn(A, 4).npu()
>>> out = npu_bbox_coder_decode_xywh2xyxy(bboxes, pred_bboxes, max_shape=(max_shape, max_shape))
>>> torch.npu.synchronize()
>>> print('npu_bbox_coder_decode_xywh2xyxy done. output shape is ', out.shape)
npu_bbox_coder_decode_xywh2xyxy done. output shape is torch.Size([1024, 4])
```

