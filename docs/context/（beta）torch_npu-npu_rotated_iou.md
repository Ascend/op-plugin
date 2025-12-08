# （beta）torch_npu.npu_rotated_iou

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

计算旋转框的IoU。

## 函数原型

```
torch_npu.npu_rotated_iou(self, query_boxes, trans=False, mode=0, is_cross=True,v_threshold=0.0, e_threshold=0.0) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：必选参数，梯度增量数据，shape为(B, 5, N)数据类型为float32的3D张量。
- **query_boxes** (`Tensor`)：必选参数，标注框，shape为(B, 5, K)数据类型为float32的3D张量，其中K值不大于1600。
- **trans** (`bool`)：可选参数，值为True表示“xyxyt”，值为False表示“xywht”。默认值为False。
- **is_cross** (`bool`)：可选参数，值为True时表示交叉计算，为False时表示一对一计算。默认值为True。
- **mode** (`int`)：可选参数，计算模式，取值为0或1。0表示IoU，1表示IoF。默认值为0。
- **v_threshold** (`float`)：可选参数，旋转框的高度阈值。默认值为0.0。
- **e_threshold** (`float`)：可选参数，旋转框的角度阈值。默认值为0.0。

## 调用示例

```python
>>> import numpy as np
>>> a=np.random.uniform(0,1,(2,2,5)).astype(np.float16)
>>> b=np.random.uniform(0,1,(2,3,5)).astype(np.float16)
>>> box1=torch.from_numpy(a).to("npu")
>>> box2=torch.from_numpy(a).to("npu")
>>> output = torch_npu.npu_rotated_iou(box1, box2, trans=False, mode=0, is_cross=True)
>>> output
tensor([[[3.3325e-01, 1.0162e-01],
        [1.0162e-01, 1.0000e+00]],

        [[0.0000e+00, 0.0000e+00],
        [0.0000e+00, 5.9605e-08]]], device='npu:0', dtype=torch.float16)
```

