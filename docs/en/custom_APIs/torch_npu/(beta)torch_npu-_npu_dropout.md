# (beta) torch\_npu.\_npu\_dropout

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas training products</term>: Supported
<!-- end id5 -->

## Function

- The dropout result is counted without using a seed.
- This API performs the dropout operation during training: it randomly sets elements in the input tensor to zero with probability p, and scales the elements that are not set to zero by 1/(1-p) to keep the expected value unchanged.

## Prototype

```python
torch_npu._npu_dropout(self, p) -> (Tensor, Tensor)
```

## Parameters

- **`self`** (`Tensor`): Required. Input tensor. The data type can be `torch.float16`, `torch.float32`, or `torch.bfloat16`. The shape supports 0 to 8 dimensions.
- **`p`** (`float`): Required. Dropout discard probability. The value range is [0, 1].

## Return Values

- **`out`** (`Tensor`): Output result of the dropout. The shape and dtype are the same as those of the input `self`.
- **`mask`** (`Tensor`): Randomly generated mask tensor. The dtype is `torch.uint8`. It is used for backward propagation.

## Constraints

- This API can be used in inference and training scenarios.
- This API supports single-operator mode.

<!-- npu="950" id6 -->
- On <term>Ascend 950PR&950DT products</term>, this API supports graph mode calls.
<!-- end id6 -->

## Example

- Single-operator call:

    ```python
    import torch
    import torch_npu
    input = torch.tensor([1.,2.,3.,4.]).npu()
    prob = 0.3
    output, mask = torch_npu._npu_dropout(input, prob)
    ```

<!-- npu="950" id7 -->
- Graph mode call: (only for <term>Ascend 950PR&950DT products</term>)

    ```python
    import torch
    import torch_npu
    import torchair
    from torchair import patch_for_hcom, get_npu_backend

    # Patch to support Hcom
    patch_for_hcom()

    # Obtain the NPU backend
    npu_backend = torchair.get_npu_backend()

    # Define a simple model
    class Model(torch.nn.Module):
        def forward(self):
            x = torch.randn(4, 8, 16, device='npu')
            m = torch.nn.Dropout(p=0.9)
            return m(x)

    # Instantiate the model and move it to the NPU
    model = Model().npu()

    # Compile the model with TorchAir
    opt_model = torch.compile(model, backend=npu_backend)

    # Execute the model
    output = opt_model()
    print("Output shape:", output.shape)
    ```
<!-- end id7 -->
