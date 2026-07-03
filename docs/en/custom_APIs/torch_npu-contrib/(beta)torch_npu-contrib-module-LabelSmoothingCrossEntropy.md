# (beta) torch_npu.contrib.module.LabelSmoothingCrossEntropy

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Performs label smoothing cross entropy loss calculation using NPU APIs.

## Prototype

```python
torch_npu.contrib.module.LabelSmoothingCrossEntropy(num_classes=1000, smooth_factor=0.)
```

## Parameters

**Computation Parameters**

- **`num_classes`** (`float`): Number of classes for one-hot encoding.
- **`smooth_factor`** (`float`): Label smoothing factor. Set this parameter to `0.1` when label smoothing is enabled. The value range of this parameter is [0, 1]. The default value is `0`.

**Computation Input**

- **`pred`** (`Tensor`): Model prediction tensor.
- **`target`** (`Tensor`): Ground truth label tensor.

## Return Values

`Tensor`

Cross entropy computation result.

## Example

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.module import LabelSmoothingCrossEntropy
>>> pred = torch.randn(2, 10).npu()
>>> target = torch.randint(0, 10, size=(2,)).npu()
>>> pred.requires_grad = True
>>> m = LabelSmoothingCrossEntropy(10)
>>> npu_output = m(pred, target)
>>> npu_output.backward()
>>> print(npu_output)
tensor(1.9443, device='npu:0', grad_fn=<MeanBackward1>)
```
