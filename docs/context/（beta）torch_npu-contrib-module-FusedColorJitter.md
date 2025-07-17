# （beta）torch_npu.contrib.module.FusedColorJitter

>**须知：**<br>
>该接口计划废弃，可以使用torchvision.transforms.ColorJitter接口进行替换。

## 函数原型

```
torch_npu.contrib.module.FusedColorJitter(torch.nn.Module)
```

## 功能说明

随机更改图像的亮度、对比度、饱和度和色调。

## 参数说明

- brightness (Float或Tuple of float (min, max)) - 亮度调整值。Brightness_factor统一从[max(0, 1 - brightness), 1 + brightness]或给定的[min, max]中选择。非负数。
- contrast (Float或Tuple of float (min, max)) - 对比度调整值。Contrast_factor统一从[max(0, 1 - contrast), 1 + contrast]或给定的[min, max]中选择。非负数。
- saturation (Float或Tuple of float (min, max)) - 饱和度调整值。Saturation_factor统一从[max(0, 1 - saturation), 1 + saturation]或给定的[min, max]中选择。非负数。
- hue (Float或Tuple of float (min, max)) - 色调调整值。Hue_factor统一从[-hue, hue]或给定的[min, max]中选择，且满足0<= hue <= 0.5或-0.5 <= min <= max <= 0.5。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> from PIL import Image 
>>> from torch_npu.contrib.module import FusedColorJitter
>>> import numpy as np
>>> image = Image.fromarray(torch.randint(0, 256, size=(224, 224, 3)).numpy().astype(np.uint8))
>>> fcj = FusedColorJitter(0.1, 0.1, 0.1, 0.1).npu()
>>> img = fcj(image)
```

