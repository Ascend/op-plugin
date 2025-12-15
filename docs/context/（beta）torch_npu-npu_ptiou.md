# （beta）torch_npu.npu_ptiou

> [!NOTICE]  
> 该接口计划废弃，可以使用`torch_npu.npu_iou`接口进行替换。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

根据ground-truth和预测区域计算交并比（IoU）或前景交叉比（IoF）。

## 函数原型

```
torch_npu.npu_ptiou(bboxes, gtboxes, mode=0) -> Tensor
```

## 参数说明

- **bboxes**（`Tensor`）：输入张量。
- **gtboxes**（`Tensor`）：输入张量。
- **mode**（`int`）：默认值为0。0为IoU模式，1为IoF模式。

## 调用示例

```python
>>> bboxes = torch.tensor([[0, 0, 10, 10],
                           [10, 10, 20, 20],
                           [32, 32, 38, 42]], dtype=torch.float16).to("npu")
>>> gtboxes = torch.tensor([[0, 0, 10, 20],
                            [0, 10, 10, 10],
                            [10, 10, 20, 20]], dtype=torch.float16).to("npu")
>>> output_iou = torch_npu.npu_iou(bboxes, gtboxes, 0)
>>> output_iou
tensor([[0.4985, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000], 
       [0.0000, 0.9961, 0.0000]], device='npu:0', dtype=torch.float16)
```

