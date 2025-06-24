# （beta）torch_npu.npu_rotated_iou

## 函数原型

```
torch_npu.npu_rotated_iou(self, query_boxes, trans=False, mode=0, is_cross=True,v_threshold=0.0, e_threshold=0.0) -> Tensor
```

## 功能说明

计算旋转框的IoU。

## 参数说明

- self (Tensor) - 梯度增量数据，shape为(B, 5, N)数据类型为float32的3D张量。
- query_boxes (Tensor) - 标注框，shape为(B, 5, K)数据类型为float32的3D张量。
- trans (Bool，默认值为False) - 值为True表示“xyxyt”，值为False表示“xywht”。
- is_cross (Bool，默认值为True) - 值为True时表示交叉计算，为False时表示一对一计算。
- mode (Int，默认值为0) - 计算模式，取值为0或1。0表示IoU，1表示IoF。
- v_threshold (Float，可选，默认值为0.0) - 旋转框的高度阈值。
- e_threshold (Float，可选，默认值为0.0) - 旋转框的角度阈值。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

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

