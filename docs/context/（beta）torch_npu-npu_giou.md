# （beta）torch_npu.npu_giou

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

首先计算两个框的最小封闭面积和IoU，然后计算封闭区域中不属于两个框的封闭面积的比例，最后从IoU中减去这个比例，得到GIoU。

## 函数原型

```
torch_npu.npu_giou(self, gtboxes, trans=False, is_cross=False, mode=0) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：必选参数，标注框，shape为(N, 4)数据类型为`float16`或`float32`的2D张量。“N”表示标注框的数量，值“4”表示[x1, y1, x2, y2]或[x, y, w, h]。
- **gtboxes** (`Tensor`)：必选参数，真值框，shape为(M, 4)数据类型为`float16`或`float32`的2D张量。“M”表示真值框的数量，值“4”表示[x1, y1, x2, y2]或[x, y, w, h]。
- **trans** (`bool`)：可选参数，值为True代表“xywh”，值为False代表“xyxy”，默认值为False。
- **is_cross** (`bool`)：可选参数，控制输出shape是[M, N]还是[1,N]。如果值为True，则输出shape为[M,N]。如果为False，则输出shape为[1,N]。默认值为False。
- **mode** (`int`)：可选参数，计算模式，取值为0或1。0表示IoU，1表示IoF。默认值为0。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> import numpy as np
>>> a=np.random.uniform(0,1,(4,10)).astype(np.float16)
>>> b=np.random.uniform(0,1,(4,10)).astype(np.float16)
>>> box1=torch.from_numpy(a).to("npu")
>>> box2=torch.from_numpy(a).to("npu")
>>> output = torch_npu.npu_giou(box1, box2, trans=True, is_cross=False, mode=0)
>>> output
tensor([[1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [1.]], device='npu:0', dtype=torch.float16)
```

