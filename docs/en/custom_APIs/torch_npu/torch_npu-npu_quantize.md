# torch_npu.npu_quantize

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
|<term>Atlas inference products</term>| √   |

## Function

- Description: Quantizes the input tensor.
- Formulas:
    - If `div_mode` is `True`:
        $$
        result=(input/scales)+zero\_points
        $$

    - If `div_mode` is `False`:

        $$
        result=(input*scales)+zero\_points
        $$

## Prototype

```python
torch_npu.npu_quantize(input, scales, zero_points, dtype, axis=1, div_mode=True) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. Source data tensor to be quantized, $input$ in the formulas. The data layout can be ND. Empty tensors are supported. Non-contiguous tensors are supported. When `div_mode` is `False` and `dtype` is `"quint4x2"`, the last dimension must be divisible by 8.
    - Atlas inference products: The data type can be `float` or `float16`.
    - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products: The data type can be `float`, `float16`, or `bfloat16`. When `div_mode` is `False` and the data type is `float`, the data layout can be NZ.

- **`scales`** (`Tensor`): Required. Scaling tensor, $scales$ in the formula. Empty tensors are supported. Non-contiguous tensors are supported.
    - When `div_mode` is `True`:
        - Atlas inference products: The data type can be `float`.
        - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products: The data type can be `float` or `bfloat16`.

    - When `div_mode` is `False`, the data layout can be ND. This parameter can be a 1D or multidimensional tensor. If it is a 1D tensor, the size of the corresponding axis must be identical to the `axis` dimension size in `input` or `1`. If it is a multidimensional tensor, its shape must match the dimension count of `input`. All dimensions except the specified `axis` dimension must be `1`, and the size of the specified `axis` dimension must be identical to the corresponding dimension size in `input` or `1`. The data type and data layout must be identical to the data type and data layout of `input`.
        - Atlas inference products: The data type can be `float` or `float16`.
        - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products: The data type can be `float`, `float16`, or `bfloat16`. When the data layout is NZ, all elements in `scales` must be identical to `1`.

- **`zero_points`** (`Tensor`): Required. Offset tensor for `input`, $zero_points$ in the formulas. This parameter can be set to `None`. Empty tensors are supported. Non-contiguous tensors are supported.
    - When `div_mode` is `True`:
        - Atlas inference products: The data type can be `int8`, `uint8`, or `int32`.
        - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products: The data type can be `int8`, `uint8`, `int32`, or `bfloat16`.

    - When `div_mode` is `False`, the data layout can be ND. This parameter can be a 1D or multidimensional tensor. If it is a 1D tensor, the size of the corresponding axis must be identical to the `axis` dimension size in `input` or `1`. If it is a multidimensional tensor, its shape must match the dimension count of `input`. All dimensions except the specified `axis` dimension must be `1`, and the size of the specified `axis` dimension must be identical to the corresponding dimension size in `input`. The shape and data type of `zero_points` must be identical to those of `scales`.
        - Atlas inference products: The data type can be `float` or `float16`.
        - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products: The data type can be `float`, `float16`, or `bfloat16`. When the data layout of `input` is NZ, this value is empty.

- **`dtype`** (`int`): Required. Data type of the output parameter.
    - When `div_mode` is `True`:
        - Atlas inference products: The data type can be `qint8`, `quint8`, or `int32`.
        - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products: The data type can be `qint8`, `quint8`, or `int32`.

    - When `div_mode` is `False`, the data type can be `qint8` or `quint4x2`. If `dtype` is `"quint4x2"`, the output tensor data type is `int32`, which is concatenated by eight `int4` values.

- **`axis`** (`int`): Optional. Quantized element-wise axis. Other axes are broadcast. The default value is `1`.

    When `div_mode` is `False`, the value range of `axis` is [-2, +∞), and the specified axis must be less than or equal to the dimension count of `input`. If `axis` is `-2`, the quantized element-wise axis is the second-to-last axis of `input`. If `axis` is greater than `-2`, the quantized element-wise axis is the last axis of `input`.

- **`div_mode`** (`bool`): Optional. Computation mode for `scales`, `div_mode` in the formulas. Valid values are `True` (`scales` is computed through division) or `False` (`scales` is computed through multiplication). The default value is `True`.

## Return Values

`Tensor`

$result$ in the formula. The data type is specified by the `dtype` parameter. If `dtype` is `"quint4x2"`, the output data type is `int32`. The last dimension of its shape is 1/8 of the last dimension of the shape of `input`, and all other dimensions must be identical to the corresponding dimensions of the shape of `input`. If `dtype` is not `"quint4x2"`, its shape must be identical to that of `input`. The output data layout must be identical to the data layout of `input`. When the data layout is NZ, the data type can only be `int32`. Empty tensors are supported. Non-contiguous tensors are supported.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.
- When the data layout of `input` is NZ, this parameter must be 3D with shape `(e, k, n)`, where `k` must be divisible by `256` and `n` must be divisible by `8`. The shape of `scales` can be 1D or 3D. The `zero_points` parameter must be set to `None`, and `dtype` must be `"quint4x2"`.
- When `div_mode` is `False`:
    - This API is supported on Atlas A2 training products/Atlas A2 inference products.
    - When `dtype` is `"quint4x2"` or `axis` is `-2`, this API is not supported on Atlas inference products.

## Examples

- Single-operator call
    - Atlas A2 training products/Atlas A2 inference products

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.randn(1, 1, 12).bfloat16().npu()
        >>> scale = torch.tensor([0.1] * 12).bfloat16().npu()
        >>> out = torch_npu.npu_quantize(x, scale, None, torch.qint8, -1, False)
        >>> print(x)
        tensor([[[ 0.9609,  1.3281, -0.6172,  0.5469, -1.1797, -1.1719, -0.7422,
                0.9727, -0.9062, -0.0815, -0.8047,  1.0703]]], device='npu:0',
            dtype=torch.bfloat16)
        >>> print(out)
        tensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], device='npu:0',
            dtype=torch.int8)
        ```

    - Atlas inference products

        ```python
        >>> import torch
        >>> import torch_npu
        >>>
        >>> x = torch.randn((2, 3, 12), dtype=torch.float).npu()
        >>> scale = torch.tensor(([3] * 12),dtype=torch.float).npu()
        >>> out = torch_npu.npu_quantize(x, scale, None, torch.qint8, -1, False)
        >>> print(x)
        tensor([[[-7.7834e-01, -1.0473e+00, -1.1155e+00,  1.2233e+00, -1.2271e+00,
                -2.5612e+00, -1.8274e-01,  2.8293e+00,  1.9029e-01, -1.9333e+00,
                -4.9270e-01, -1.0650e+00],
                [-8.9416e-01,  3.1869e-02, -5.8144e-01, -4.9477e-01,  9.7323e-02,
                -3.8681e-01,  2.1969e-03, -6.3244e-01,  7.1591e-01, -1.8587e-01,
                -1.3381e+00, -2.6253e-01],
                [ 1.8462e-02,  1.2397e-01, -9.0656e-01, -9.9280e-01, -4.4235e-02,
                1.0623e+00, -9.8437e-02,  1.2941e+00,  1.0805e+00, -1.7269e-01,
                -9.9205e-02, -6.1429e-01]],

                [[ 1.3678e+00, -2.7348e-01, -4.1354e-01, -9.4638e-01,  4.2792e-01,
                8.0462e-01,  9.3584e-01,  6.3704e-01,  1.1269e+00, -1.5329e+00,
                5.8572e-01, -1.3966e+00],
                [ 3.5882e-01,  8.7029e-01, -1.3176e+00,  1.1601e+00, -3.6984e-01,
                7.3642e-01, -1.0755e+00,  6.6557e-01,  3.1149e+00, -6.8776e-01,
                -1.0913e+00,  4.4962e-01],
                [-1.2505e+00,  1.5474e+00, -7.4332e-02, -1.6657e+00,  1.3275e+00,
                5.8914e-02,  8.4287e-01, -1.7109e+00,  1.8256e-01,  3.2937e-01,
                2.4875e+00,  1.3921e+00]]], device='npu:0')
        >>> print(out)
        tensor([[[-2, -3, -3,  4, -4, -8, -1,  8,  1, -6, -1, -3],
                [-3,  0, -2, -1,  0, -1,  0, -2,  2, -1, -4, -1],
                [ 0,  0, -3, -3,  0,  3,  0,  4,  3, -1,  0, -2]],

                [[ 4, -1, -1, -3,  1,  2,  3,  2,  3, -5,  2, -4],
                [ 1,  3, -4,  3, -1,  2, -3,  2,  9, -2, -3,  1],
                [-4,  5,  0, -5,  4,  0,  3, -5,  1,  1,  7,  4]]], device='npu:0',
            dtype=torch.int8)
        ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig

    x = torch.randn((2, 3, 12), dtype=torch.float16).npu()
    scale = torch.tensor(([3] * 12), dtype=torch.float16).npu()
    axis = 1
    div_mode = False


    class Network(torch.nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, x, scale, zero_points, dst_type, div_mode):
            return torch_npu.npu_quantize(x, scale, zero_points=zero_points, dtype=dst_type, div_mode=div_mode)


    model = Network()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    config.debug.graph_dump.type = 'pbtxt'
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=True)
    output_data = model(x, scale, None, dst_type=torch.qint8, div_mode=div_mode)
    print("shape of x:", x.shape)
    print("shape of output_data:", output_data.shape)
    print("x:", x)
    print("output_data:", output_data)



    # Expected output of the preceding code sample:
    shape of x: torch.Size([2, 3, 12])
    shape of output_data: torch.Size([2, 3, 12])
    x: tensor([[[ 0.2600,  0.6782,  0.5024,  0.9492,  0.6089,  0.7461,  1.5332,
            -0.2123,  0.6558, -0.8354, -0.5366, -0.6821],
            [-0.2522,  0.2415, -0.0269, -0.1497,  0.2256, -0.5239,  0.7363,
            -0.2468,  1.6064,  1.4170, -0.2213,  1.5947],
            [-0.6328,  0.8105,  0.2532,  1.0684, -1.2119, -0.6865,  0.7451,
            -0.8120,  0.6401, -2.1270, -0.9482, -1.1973]],

            [[-1.7461, -1.1758, -0.5352,  1.5938,  1.8945, -2.2500, -0.5073,
            -0.8164,  0.8267, -0.4377,  1.2490,  0.2415],
            [ 0.8062, -1.0498, -0.8345,  1.1465, -0.7349,  0.1317,  0.2280,
            -0.8145,  0.2673,  1.4756, -1.6768,  1.1572],
            [-0.3147, -0.4446, -1.0508,  0.8325,  1.4590,  0.2096, -0.9961,
            0.6089, -0.2460,  1.1543,  0.9277,  0.1079]]], device='npu:0',
        dtype=torch.float16)
        
    output_data: tensor([[[ 1,  2,  2,  3,  2,  2,  5, -1,  2, -3, -2, -2],
            [-1,  1,  0,  0,  1, -2,  2, -1,  5,  4, -1,  5],
            [-2,  2,  1,  3, -4, -2,  2, -2,  2, -6, -3, -4]],

            [[-5, -4, -2,  5,  6, -7, -2, -2,  2, -1,  4,  1],
            [ 2, -3, -3,  3, -2,  0,  1, -2,  1,  4, -5,  3],
            [-1, -1, -3,  2,  4,  1, -3,  2, -1,  3,  3,  0]]], device='npu:0',
        dtype=torch.int8)

    ```
