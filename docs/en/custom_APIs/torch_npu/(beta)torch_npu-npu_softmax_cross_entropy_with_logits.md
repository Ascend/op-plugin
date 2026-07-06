# (beta) torch_npu.npu_softmax_cross_entropy_with_logits

> [!NOTICE]
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>    |     √    |
|<term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

- Description: Computes the cross-entropy loss for softmax.

- Formula:

$$
     loss = -\sum_{i=1}^{N}y_i * log(softmax(x_i))
$$
$x_i$ corresponds to the input `features`, $y_i$ corresponds to the input `labels`, and $N$ indicates the length of the input features.

## Prototype

```python
torch_npu.npu_softmax_cross_entropy_with_logits(features, labels) -> Tensor
```

## Parameters

- **`features`** (`Tensor`): Required. Input features, $x_i$ in the formula. This parameter must be a matrix with shape `(1, batch_size * num_classes)`. The data layout can be ND. Non-contiguous tensors are supported. This parameter can be up to 2D. Empty tensors are supported. The data type can be `float`, `float16`, or `bfloat16`.

  - Atlas training products and Atlas inference products: The `bfloat16` data type is not supported.

- **`labels`** (`Tensor`): Required. Input labels, $y_i$ in the formula. The shape and data type must be identical to those of `features`. The data layout can be ND. Non-contiguous tensors are supported. This parameter can be up to 2D. Empty tensors are supported. The data type can be `float`, `float16`, or `bfloat16`.

  - Atlas training products and Atlas inference products: The `bfloat16` data type is not supported.

## Return Values

`Tensor`

Output tensor representing the cross-entropy loss computation results of softmax and cross entropy, $loss$ in the formula.

## Example

```python
>>> import torch, torch_npu
>>> batch_size = 4
>>> num_classes = 12
>>> features = torch.rand(1, batch_size * num_classes).npu()
>>> labels = torch.rand(1, batch_size * num_classes).npu()
>>> output = torch_npu.npu_softmax_cross_entropy_with_logits(features, labels)
>>> print(output)
tensor([97.9450], device='npu:0')
```
