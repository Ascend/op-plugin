# (beta) torch_npu.contrib.module.FusedColorJitter

> [!NOTICE]  
> This API is planned for deprecation. Use `torchvision.transforms.ColorJitter` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Randomly adjusts the brightness, contrast, saturation, and hue of an image.

## Prototype

```python
torch_npu.contrib.module.FusedColorJitter(torch.nn.Module)
```

## Parameters

- **`brightness`** (`float` or `Tuple[float, float]`): Brightness adjustment range factor. `brightness_factor` is selected from `[max(0, 1 - brightness), 1 + brightness]` or the specified `[min, max]`. The value must be non-negative.
- **`contrast`** (`float` or `Tuple[float, float]`): Contrast adjustment range factor. `contrast_factor` is selected from `[max(0, 1 - contrast), 1 + contrast]` or the specified `[min, max]`. The value must be non-negative.
- **`saturation`** (`float` or `Tuple[float, float]`): Saturation adjustment range factor. `saturation_factor` is selected from `[max(0, 1 - saturation), 1 + saturation]` or the specified `[min, max]`. The value must be non-negative.
- **`hue`** (`float` or `Tuple[float, float]`): Hue adjustment range factor. `hue_factor` is selected from `[-hue, hue]` or the specified `[min, max]`. The value must satisfy `0 <= hue <= 0.5` or `-0.5 <= min <= max <= 0.5`.

## Example

```python
>>> import torch
>>> from PIL import Image 
>>> from torch_npu.contrib.module import FusedColorJitter
>>> import numpy as np
>>> image = Image.fromarray(torch.randint(0, 256, size=(224, 224, 3)).numpy().astype(np.uint8))
>>> fcj = FusedColorJitter(0.1, 0.1, 0.1, 0.1).npu()
>>> img = fcj(image)
```
