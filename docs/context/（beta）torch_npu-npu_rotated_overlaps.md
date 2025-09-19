# （beta）torch_npu.npu_rotated_overlaps

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

计算旋转框的重叠面积。

## 函数原型

```
torch_npu.npu_rotated_overlaps(self, query_boxes, trans=False) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：必选参数，梯度增量数据，shape为(B, 5, N)数据类型为`float32`的3D张量。
- **query_boxes** (T`ensor`)：必选参数，标注框，shape为(B, 5, K)数据类型为`float32`的3D张量。
- **trans** (`bool`)：可选参数，值为True表示“xyxyt”，值为False表示“xywht”。默认值为False。

## 调用示例

```python
>>> import numpy as np
>>> a=np.random.uniform(0,1,(1,3,5)).astype(np.float16)
>>> b=np.random.uniform(0,1,(1,2,5)).astype(np.float16)
>>> box1=torch.from_numpy(a).to("npu")
>>> box2=torch.from_numpy(a).to("npu")
>>> output = torch_npu.npu_rotated_overlaps(box1, box2, trans=False)
>>> output
tensor([[[0.0000, 0.1562, 0.0000],
        [0.1562, 0.3713, 0.0611],
        [0.0000, 0.0611, 0.0000]]], device='npu:0', dtype=torch.float16)
```

