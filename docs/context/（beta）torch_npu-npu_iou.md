# （beta）torch_npu.npu_iou

## 函数原型

```
torch_npu.npu_iou(bboxes, gtboxes, mode=0) -> Tensor 
```

## 功能说明

根据ground-truth和预测区域计算交并比（IoU）或前景交叉比（IoF）。

## 参数说明

- bboxes (Tensor) - 输入张量。
- gtboxes (Tensor) - 输入张量。
- mode (Int，默认值为0) - 0为IoU模式，1为IoF模式。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

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

