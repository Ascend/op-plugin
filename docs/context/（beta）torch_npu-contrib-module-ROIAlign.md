# （beta）torch_npu.contrib.module.ROIAlign

## 函数原型

```
torch_npu.contrib.module.ROIAlign(nn.Module)
```

## 功能说明

使用NPU API进行ROIAlign。

## 参数说明

- output_size (Tuple) - h, w。
- spatial_scale (Float) - 按此参数值缩放输入框。
- sampling_ratio (Int) - 为每个输出样本采集的输入样本数。0表示密集采样。
- aligned (Bool) - 如果值为False，使用Detectron中的原实现方式。如果值为True，可更准确地对齐结果。

    >**说明：**<br>
    >aligned=True含义：
    >给定一个连续坐标c，使用floor(c - 0.5)和ceil(c - 0.5)对它的两个相邻像素索引（像素模型中）进行计算。例如，c=1.3具有离散索引为[0]和[1] （从连续坐标0.5到1.5的底层信号采样）的像素邻域。但原始ROIAlign（aligned=False）在计算相邻像素索引时不会减去0.5，因此在执行双线性插值时，它使用的是未完全对齐的像素（相对于我们的像素模型有一点不对齐）。当aligned=True，首先适当缩放ROI，然后在调用ROIAlign之前将其移动-0.5。这样可以生成正确的邻域。相关验证请参见[detectron2/tests/testroialign.py](https://github.com/facebookresearch/detectron2/blob/v0.2/tests/layers/test_roi_align.py)。如果ROIAlign与conv层一起使用，差异也不会对模型的性能产生影响。

## 输出说明

Float - shape为(k, 5)和(k, 1)的张量。标签以0为基础。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> from torch_npu.contrib.module import ROIAlign
>>> input1 = torch.FloatTensor([[[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36]]]]).npu()
>>> roi = torch.tensor([[0, -2.0, -2.0, 22.0, 22.0]]).npu()
>>> output_size = (3, 3)
>>> spatial_scale = 0.25
>>> sampling_ratio = 2
>>> aligned = False
>>> input1.requires_grad = True
>>> roi.requires_grad = True
>>> model = ROIAlign(output_size, spatial_scale, sampling_ratio, aligned=aligned).npu()
>>> output = model(input1, roi)
>>> output.sum().backward()
```

