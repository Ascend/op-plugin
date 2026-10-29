# torch_npu.npu_grouped_matmul_swiglu_quant

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->

## Function

<!-- npu="A3,910b" id3 -->
- API function: Fuses four computation stages: GroupedMatmul, dequant, swiglu (SwiGLU activation), and quant. It is used by the DeepSeek model to optimize performance compared with small operators. The weight must be passed in FRACTAL_NZ format. This API is a simplified version of `npu_grouped_matmul_swiglu_quant_v2` and supports only <term>Atlas A3 products</term> and <term>Atlas A2 products</term>.
  
  Multi-operator fusion computation order (executed in sequence):
  1. GroupedMatmul: Split tokens into groups according to group_list, and compute $X_{i}\cdot W_{i}$ for each group.
  2. dequant: Multiply the matrix multiplication result element-wise by the activation quantization factor $x\_scale$ and the weight quantization factor $w\_scale$ to complete dequantization.
  3. swiglu: Split the dequantization result in half along the N axis into $C_{i,act}$ and $gate_{i}$, and compute $S_{i}=Swish(C_{i,act})\odot gate_{i}$, where $Swish(x)=\frac{x}{1+e^{-x}}$.
  4. quant: $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$, $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$ to obtain the quantized output Q and the quantization factor Q_scale.
<!-- end id3 -->
- Quantization scenario **A8W8** (A indicates the activation matrix, W indicates the weight matrix, and 8 indicates the `torch.int8` data type):
  - Input:
    - $X∈\mathbb{Z_8}^{M \times K}$ is the activation matrix (left matrix), where M is the total number of tokens and K is the feature dimension.
    - $W∈\mathbb{Z_8}^{E \times K \times N}$ is the grouped weight matrix (right matrix), where E is the number of experts and N is the output dimension.
    - $w\_scale∈\mathbb{R}^{E \times N}$ is the per-channel scaling factor of the weight.
    - $x\_scale∈\mathbb{R}^{M}$ is the per-token scaling factor of the activation.
    - $groupList∈\mathbb{N}^{E}$ is the cumsum group index list.
  - Output:
    - $Q∈\mathbb{Z_8}^{M \times N / 2}$ is the quantized output matrix.
    - $Q\_scale∈\mathbb{R}^{M}$ is the quantization scaling factor.
  - Computation process:

    $$
    C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\,\text{Broadcast}} \odot w\_scale_{i\,\text{Broadcast}}
    $$

    $$
    C_{i,act}, gate_{i} = split(C_{i})
    $$

    $$
    S_{i}=Swish(C_{i,act})\odot gate_{i}
    $$

  - Quantized output:

    $$
    Q\_scale_{i} = \frac{max(|S_{i}|)}{127}
    $$

    $$
    Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil
    $$

- Quantization scenario **A8W4** (A indicates the activation matrix, W indicates the weight matrix, and 4 indicates the `torch_npu.int4` data type): The activation is `torch.int8` and the weight is `torch_npu.int4`. The left matrix is split into high and low 4-bit parts, which are separately matrix-multiplied with the weight. The result is then restored by using the offline-generated auxiliary matrix bias. The rest of the computation process is the same as that of A8W8. In this scenario, `weight_scale` supports 2D (perchannel) or 3D (pergroup) input.
  - Input:
    - $X∈\mathbb{Z_8}^{M \times K}$: Activation matrix (left matrix), where M is the total number of tokens and K is the feature dimension.
    - $W∈\mathbb{Z_4}^{E \times K \times N}$: Grouped weight matrix (right matrix), where E is the number of experts, K is the feature dimension, and N is the output dimension.
    - $weightAssistMatrix∈\mathbb{R}^{E \times N}$: Auxiliary matrix used during matrix multiplication (for the calculation process of the auxiliary matrix, see below).
    - $w\_scale∈\mathbb{R}^{E \times K\_group\_num \times N}$: Per-channel scaling factor of the grouped weight matrix (right matrix), where E is the number of experts, K_group_num is the number of groups on the K axis, and N is the output dimension.
    - $x\_scale∈\mathbb{R}^{M}$: Per-token scaling factor of the activation matrix (left matrix), where M is the total number of tokens.
    - $groupList∈\mathbb{N}^{E}$: cumsum group index list.
  - Output:
    - $Q∈\mathbb{Z_8}^{M \times N / 2}$: Quantized output matrix.
    - $Q\_scale∈\mathbb{R}^{M}$: Quantization scaling factor.
  - Computation process:
    1. Calculation process for generating the auxiliary matrix (weightAssistMatrix):
        - perchannel quantization:

          $$
          weightAssistMatrix_{i} = 8 × w\_scale × Σ_{k=0}^{K-1} weight[:,k,:]
          $$

        - pergroup quantization ($w\_scale$ is 3D):
  
          $$
          weightAssistMatrix_{i} = 8 × Σ_{k=0}^{K-1} (weight[:,k,:] × w\_scale[:, \lfloor\frac{k}{num\_per\_group}\rfloor, :])
          $$

          where

          $$ num\_per\_group = K // K\_group\_num $$

    2. Convert the left matrix $\mathbb{Z_8}$ into high and low 4-bit parts $\mathbb{Z_4}$:

        $$
        X\_high\_4bits_{i} = \left\lfloor \frac{X_{i}}{16} \right\rfloor
        $$

        $$
        X\_low\_4bits_{i} = And(X_{i}, \mathrm{0x0f}) - 8
        $$

    3. Enable perchannel or pergroup quantization during matrix multiplication.
        - perchannel quantization:

          $$
          C\_high_{i} = (X\_high\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}
          $$

          $$
          C\_low_{i} = (X\_low\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}
          $$

        - pergroup quantization:

          $$
          C\_high_{i} = \\ Σ_{k=0}^{K-1}((X\_high\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )
          $$

          $$
          C\_low_{i} = \\ Σ_{k=0}^{K-1}((X\_low\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )
          $$

    4. Restore the high and low matrix multiplication results to the overall result:

        $$
        C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAssistMatrix_{i}) \odot x\_scale_{i}
        $$

        $$
        C_{i,act}, gate_{i} = split(C_{i})
        $$

        $$
        S_{i}=Swish(C_{i,act})\odot gate_{i}
        $$

        where $Swish(x)=\frac{x}{1+e^{-x}}$.

  - Quantized output:

    $$
    Q\_scale_{i} = \frac{max(|S_{i}|)}{127}
    $$

    $$
    Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil
    $$

- group_list grouping description: The tokens of the current group are determined according to groupList[i], where $i \in [0,Len(groupList))$. For example, if groupList=[3,4,4,6]: the 0th right matrix W[0,:,:] corresponds to tokens x[0:3]; the 1st corresponds to x[3:4]; the 2nd corresponds to x[4:4] (0 tokens); and the 3rd corresponds to x[4:6]. The part not specified in groupList does not participate in the update. For example, if groupList=[12,14,18] and the shape of x is [30,N/2], Q[18:, :] and Q_scale[18:] are not updated or initialized, and the data is the original data when the device memory is allocated. That is, the output Q[:groupList[-1],:] and Q_scale[:groupList[-1]] are the valid data parts.

> [!NOTE]
>
> - The weight must first be converted to FRACTAL_NZ format (5D) by using `torch_npu.npu_format_cast(weight, 29)` before being passed. This API ignores the data layout flag of the weight and directly forces it to be treated as FRACTAL_NZ format, but the storage shape of the weight must be the 5D shape after NZ packing.
> - This API provides non-in-place semantics: the output is a newly created Tensor, and the input is not modified.

## Prototype

```python
torch_npu.npu_grouped_matmul_swiglu_quant(x, weight, group_list, weight_scale, x_scale, *, bias=None, offset=None) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Left matrix, $X$ in the formulas. The shape can be 2D [M, K]. The data type can be `torch.int8`. The data layout can be ND. Non-contiguous Tensors are supported. In the A8W8 scenario, K must be less than 65536. In the A8W4 scenario, K must be less than 20000.
- **`weight`** (`Tensor`): Required. Weight matrix, $W$ in the formulas. It must first be converted to FRACTAL_NZ format (5D storage shape) by using `torch_npu.npu_format_cast(weight, 29)`. The data type can be `torch.int8`, `torch_npu.int4`, or `torch.int32` (`torch.int32` is used for adaptation; actually, one `torch.int32` is interpreted as eight `torch_npu.int4` values). Non-contiguous Tensors are supported. This API ignores the data layout of the weight and forces it to be treated as FRACTAL_NZ format.
- **`group_list`** (`Tensor`): Required. Indicates the number of tokens participating in the computation in each group, $groupList$ in the formulas. The shape is 1D, and the length must be equal to the first-axis dimension of the weight (the number of experts E). The data type can be `torch.int64`. The data layout can be ND. Non-contiguous Tensors are supported. It is a group index list in cumsum form, and the last value constrains the valid part of the output data.
- **`weight_scale`** (`Tensor`): Required. Quantization factor of the right matrix, $w\_scale$ in the formulas. The shape can be 2D (perchannel) or 3D (pergroup). The first-axis length must be equal to the first-axis dimension of the weight, and the last-axis length must be the same as the last axis (N) after the weight is restored to ND format. The data type can be `torch.float32`, `torch.float16`, `torch.bfloat16`, or `torch.int64` (perchannel). The data layout can be ND. Non-contiguous Tensors are supported.
- **`x_scale`** (`Tensor`): Required. Quantization factor of the left matrix, $x\_scale$ in the formulas. The shape is 1D, and the length must be equal to the first-axis dimension of x (M). The data type can be `torch.float32`. The data layout can be ND. Non-contiguous Tensors are supported.
- **`*`**: Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.
- **`bias`** (`Tensor`): Optional. Auxiliary matrix during matrix multiplication, $weightAssistMatrix$ in the formulas. The shape can be 2D [E, N]. The data type can be `torch.float32`. The data layout can be ND. It takes effect only in A8W4 scenarios. In A8W8 scenarios, pass None. The default value is None.
- **`offset`** (`Tensor`): Optional. Offset for perchannel asymmetric dequantization. Reserved input. It is not supported currently. Pass None. The default value is None.

## Return Values

- **`output`** (`Tensor`): Quantized output, $Q$ in the formulas. The data type can be `torch.int8`. The shape can be 2D [M, N/2]. The data layout can be ND. group_list specifies the valid value range in the input and output. The valid data ends at group_list[-1], that is, `output[:group_list[-1], :]` is valid data and the rest is not updated.
- **`output_scale`** (`Tensor`): Output quantization factor, $Q\_scale$ in the formulas. The data type can be `torch.float32`. The shape can be 1D [M]. The data layout can be ND. The valid data ends at group_list[-1], that is, `output_scale[:group_list[-1]]` is valid data.
- **`output_offset`** (`Tensor`): Offset for output asymmetric quantization. Reserved output. It is not used currently. An empty Tensor (0D `torch.float32`) is returned.

## Constraints

- This API supports training and inference scenarios.
- Deterministic computation: The deterministic implementation is used by default.
- **A8W8** scenario:
  - The last-axis length (K) of x must not be greater than or equal to 65536.
  - The N-axis length must not exceed 10240.
  - bias must be passed as None.
- **A8W4** scenario:
  - The last-axis length (K) of x must not be greater than or equal to 20000.
  - The N-axis length must not exceed 10240.
  - bias (weightAssistMatrix) is required.
- The weight must be stored in 5D FRACTAL_NZ format, and the first-axis length must be the number of experts E.
- The length of group_list must be equal to the first-axis dimension of the weight. group_list is in cumsum form, and its last value constrains the valid data range of the output.

## Examples

- Single-operator call (A8W8 scenario)

    ```python
    import torch
    import torch_npu
    import numpy as np

    torch.npu.config.allow_internal_format = True  # This must be enabled; otherwise, the FRACTAL_NZ format cannot be passed correctly.

    def generate_non_decreasing_sequence(length, upper_limit):
        # Generate random increments.
        random_increments = torch.randint(1, 128, (length,), dtype=torch.int64)  # Avoid zero increments.
        # Accumulate to generate a non-decreasing sequence.
        sequence = torch.cumsum(random_increments, dim=0)
        # Ensure that the last element does not exceed the upper limit.
        if sequence[-1] > upper_limit:
            # Scale linearly to ensure that the sum does not exceed the upper limit.
            scale_factor = upper_limit / sequence[-1].item()
            sequence = (sequence * scale_factor).to(torch.int64)
            for i in range(1, length):
                if sequence[i] <= sequence[i-1]:
                    sequence[i] = sequence[i-1] + 1
        return sequence

    def gen_input_data(E=16, M=512, K=7168, N=4096):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8).npu()
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8).npu()
        # Convert to the 5D FRACTAL_NZ storage shape (E, N//32, K//16, 16, 32).
        weight_npu = torch_npu.npu_format_cast(weight, 29).view(E, N // 32, K // 16, 16, 32)
        weight_scale = torch.randn(E, N, dtype=torch.float32).npu()
        x_scale = torch.randn(M, dtype=torch.float32).npu()
        group_list = generate_non_decreasing_sequence(E, M).npu()
        output, output_scale, output_offset = torch_npu.npu_grouped_matmul_swiglu_quant(x, weight_npu, group_list, weight_scale, x_scale,bias=None, offset=None)
        # Valid data is group_list[-1].
        valid_output = output[:group_list[-1], :]
        valid_output_scale = output_scale[:group_list[-1]]
        return output, output_scale, output_offset

    if __name__ == "__main__":
        output, output_scale, output_offset = gen_input_data()
        print(output.shape, output.dtype)
        print(output_scale.shape, output_scale.dtype)
        print(output_offset.shape) 
    ```

    The output is as follows:

    ```text
    torch.Size([512, 2048]) torch.int8
    torch.Size([512]) torch.float32
    torch.Size([])
    ```

- Single-operator call (A8W4 scenario)

    ```python
    import torch
    import torch_npu
    import numpy as np

    torch.npu.config.allow_internal_format = True  # This must be enabled; otherwise, the FRACTAL_NZ format cannot be passed correctly.

    def generate_non_decreasing_sequence(length, upper_limit):
        # Generate random increments.
        random_increments = torch.randint(1, 128, (length,), dtype=torch.int64)  # Avoid zero increments.
        # Accumulate to generate a non-decreasing sequence.
        sequence = torch.cumsum(random_increments, dim=0)
        # Ensure that the last element does not exceed the upper limit.
        if sequence[-1] > upper_limit:
            # Scale linearly to ensure that the sum does not exceed the upper limit.
            scale_factor = upper_limit / sequence[-1].item()
            sequence = (sequence * scale_factor).to(torch.int64)
            for i in range(1, length):
                if sequence[i] <= sequence[i-1]:
                    sequence[i] = sequence[i-1] + 1
        return sequence

    def gen_input_data(E=16, M=512, K=7168, N=4096):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8).npu()
        # A8W4: The weight is int4. Every two int4 values along the N axis are packed into one int8, so the shape of the packed weight is (E, K, N//2).
        weight_pack = torch.randint(-128, 127, (E, K, N // 2), dtype=torch.int8).npu()
        # Convert to the 5D FRACTAL_NZ storage shape (E, N//64, K//16, 16, 8).
        weight_npu = torch_npu.npu_format_cast(weight_pack, 29).view(torch.int32).view(E, N // 64, K // 16, 16, 8)
        # perchannel scenario: weight_scale is int64 with shape (E, N). pergroup scenario: the shape is (E, k_group_num, N).
        weight_scale = torch.randint(-100, 100, (E, N), dtype=torch.int64).npu()
        x_scale = torch.randn(M, dtype=torch.float32).npu()
        # In the A8W4 scenario, bias (weightAssistMatrix) is required, with shape (E, N) and data type float32.
        bias = torch.randn(E, N, dtype=torch.float32).npu()
        group_list = generate_non_decreasing_sequence(E, M).npu()
        output, output_scale, output_offset = torch_npu.npu_grouped_matmul_swiglu_quant(
            x, weight_npu, group_list, weight_scale, x_scale, bias=bias, offset=None)
        # Valid data ends at group_list[-1].
        valid_output = output[:group_list[-1], :]
        valid_output_scale = output_scale[:group_list[-1]]
        return output, output_scale, output_offset

    if __name__ == "__main__":
        output, output_scale, output_offset = gen_input_data()
        print(output.shape, output.dtype)
        print(output_scale.shape, output_scale.dtype)
        print(output_offset.shape)
    ```

    The output is as follows:

    ```text
    torch.Size([512, 2048]) torch.int8
    torch.Size([512]) torch.float32
    torch.Size([])
    ```
