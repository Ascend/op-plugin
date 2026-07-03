# torch\_npu.npu\_all\_to\_all\_matmul

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |

## Function

- Description: Fuses `AlltoAll` communication, `Permute` (to ensure contiguous memory addresses after communication), and `Matmul` computation using a **communication-before-computation** sequence.

- Formulas: Assume that the input `x1` shape is `(BS, H)`, and `rankSize` represents the number of NPUs.

  $$
  commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
  permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
  output = permutedOut @ x2 + bias \\
  $$

## Prototype

```python
torch_npu.npu_all_to_all_matmul(x1, x2, hcom, world_size, bias=None, all2all_axes=None, all2all_out_flag=True) -> (Tensor, Tensor)
```

## Parameters

- **`x1`** (`Tensor`): Required. Left matrix input of the fused operator and the left operand of MatMul computation, $x1$ in the formulas. This parameter must be a 2D tensor, with shape `(BS, H)`. The data type can be `bfloat16` or `float16`. The data layout is ND. Non-contiguous tensors are not supported. Empty tensors where the first dimension size is `0` are supported.
- **`x2`** (`Tensor`): Required. Right matrix input of the fused operator and the right operand of MatMul computation, $x2$ in the formulas. This parameter must be a 2D tensor, with shape `(H*rankSize, N)`. The data type must be identical to that of `x1`. The data layout can be ND. Transposed non-contiguous tensors are supported.
- **`hcom`** (`str`): Required. Communication domain identifier string on the host side, representing the communication domain name obtained using the `get_hccl_comm_name` API.
- **`world_size`** (`int`): Required. Total number of ranks within the communication domain, $rankSize$ in the formulas. Valid values are `2`, `4`, `8`, or `16`.
- **`bias`** (`Tensor`): Optional. Accumulative bias added after the matrix multiplication computation, $bias$ in the formulas. When the data type of `x1` and `x2` is `float16`, the data type of `bias` is `float16`. When the data type of `x1` and `x2` is `bfloat16`, the data type of `bias` is `float32`. This parameter must be a 1D tensor, with shape `(N,)`. The data layout can be ND.
- **`all2all_axes`** (`List[int]`): Optional. Data exchange axis configuration for the AlltoAll and Permute operations. This parameter can be empty or be set to `[-2, -1]`, indicating that the MatMul computation result is converted from `(BS, H)` to `(BS/rankSize, H*rankSize)`.
- **`all2all_out_flag`** (`bool`): Optional. Specifies whether to output the result after `AlltoAll` and `Permute` operations. The default value is `True`.

## Return Values

- **`y`** (`Tensor`): Computation output tensor representing the final computation result. This parameter must be a 2D tensor, with shape `(BS/rankSize, N)`. The data type is identical to that of `x1` or `x2`. The data layout can be ND. Non-contiguous tensors are not supported.
- **`all2all_out`** (`Tensor`): The output tensor after the `AlltoAll` and `Permute` operations, $permutedOut$ in the formulas. When `all2all_out_flag` is `True`, the actual tensor is returned. The data type must be identical to `x1` or `x2`. The shape must be 2D with shape `(BS/rankSize, H*rankSize)`. The data layout can be ND. Non-contiguous tensors are not supported.

## Constraints

- This API can be used in training and inference scenarios.
- In A3 scenarios, this API supports single-operator execution mode and does not support graph mode.
- Except for `x1`, empty tensors are not supported for all input parameters.
- Empty strings are not supported for the communication domain name `hcom`, and its length must be in the range [1, 127].
- Variables used in input parameter tensor shapes:
    - `BS`: Size of the first dimension of the left input matrix, representing the number of input sequences. The value range is [0, 2147483647], and the value must be divisible by `rankSize`.
    - `H`: Size of the second dimension of the left input matrix, representing the hidden layer dimension. The value range is limited by the number of NPUs. The value range of `H*rankSize` is [2, 65535].
    - `H*rankSize`: Size of the first dimension of the right input matrix, representing the hidden layer dimension after AlltoAll communication of the left input matrix. The value range is [2, 65535].
    - `N`: Size of the second dimension of the right input matrix, representing the sequence length of the output sequence. The value range is [1, 2147483647].

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp

    def run_npu_all_to_all_matmul(rank, world_size, master_ip, master_port, x1_shape, x2_shape):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
        x1_tensor = torch.randn(x1_shape, dtype=torch.float16).npu() 
        x2_tensor = torch.randn(x2_shape, dtype=torch.float16).npu()    
        output, all2allout = torch_npu.npu_all_to_all_matmul(
            x1_tensor,
            x2_tensor,
            hcom_info,
            world_size
        )
        print("output: ", output)
        print("all2all_out: ", all2allout)

    if __name__ == "__main__":
        worksize = 2 # Number of NPUs
        master_ip = '127.0.0.1' # Communication IP address
        master_port = '50001' # Communication port
        x1_shape = [1024, 256] # Input shape of x1
        x2_shape = [512, 3072] # Input shape of x2
        mp.spawn(
            run_npu_all_to_all_matmul,
            args=(worksize, master_ip, master_port, x1_shape, x2_shape),
            nprocs=worksize,
        )
    ```
