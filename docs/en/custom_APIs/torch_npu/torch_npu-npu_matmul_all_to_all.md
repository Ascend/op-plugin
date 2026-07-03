# torch\_npu.npu\_matmul\_all\_to\_all

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |

## Function

- Description: Fuses matrix multiplication (MatMul) computation, data transposition (Permute) to ensure a contiguous memory data layout after communication, and AlltoAll collective communication. **Computation is performed before communication**.

- Formulas: (Assume that the shape of `x1` is `(BS, H1)`, the shape of `x2` is `(H1, H2)`, and `rankSize` indicates the number of NPU processors.)

  $$
  computeOut = x1 @ x2 + bias \\
  permutedOut = computeOut.view(BS, rankSize, H2/rankSize).permute(1, 0, 2) \\
  output = AlltoAll(permutedOut).view(rankSize*BS, H2/rankSize)
  $$

## Prototype

```python
torch_npu.npu_matmul_all_to_all(x1, x2, hcom, world_size, bias=None, all2all_axes=None) -> Tensor
```

## Parameters

- **`x1`** (`Tensor`): Required. Left matrix input of the fused operator and the left operand of MatMul computation, $x1$ in the formulas. This parameter must be 2D with shape `(BS, H1)`. The data type can be `bfloat16` or `float16`. The data layout is ND. Non-contiguous tensors are not supported. Empty tensors where the first dimension size is `0` are supported.
- **`x2`** (`Tensor`): Required. Right matrix input of the fused operator and the right operand of MatMul computation, $x2$ in the formulas. This parameter must be 2D with shape `(H1, H2)`. The data type must be identical to that of `x1`. The data layout is ND. Transposed non-contiguous tensors are supported.
- **`hcom`** (`str`): Required. Communication domain identifier string on the host side, representing the communication domain name obtained using the `get_hccl_comm_name` API.
- **`world_size`** (`int`): Required. Total number of ranks within the communication domain, $rankSize$ in the formulas. Valid values are `2`, `4`, `8`, or `16`.
- **`bias`** (`Tensor`): Optional. Accumulative bias added after the matrix multiplication computation, $bias$ in the formulas. When the data type of `x1` and `x2` is `float16`, the data type of `bias` is `float16`. When the data type of `x1` and `x2` is `bfloat16`, the data type of `bias` is `float32`. This parameter must be 1D with shape `(H2,)`. The data layout can be ND.
- **`all2all_axes`** (`List[int]`): Optional. Data exchange axis configuration for the AlltoAll and Permute operations. This parameter can be empty or be set to `[-1, -2]`, indicating that the MatMul computation result is converted from `(BS, H2)` to `(BS * rankSize, H2/rankSize)`.

## Return Values

- **`y`** (`Tensor`): Computation output tensor representing the final computation result `output`. This parameter must be 2D with shape `(BS * rankSize, H2/rankSize)`. The data type is identical to that of `x1` or `x2`. The data layout can be ND. Non-contiguous tensors are not supported.

## Constraints

- This API can be used in training and inference scenarios.
- In A3 scenarios, this API supports single-operator execution mode and does not support graph mode.
- Except for `x1`, empty tensors are not supported for all input parameters.
- Empty strings are not supported for the communication domain name `hcom`, and its length must be in the range [1, 127].
- Variables used in input parameter tensor shapes:
    - `BS` indicates the size of the first dimension of the input left matrix, representing the count of sequences in the input sequence. The value range of $BS*rankSize$ is [0, 2147483647].
    - `H1` indicates the size of the second dimension of the input left matrix and the size of the first dimension of the input right matrix, representing the hidden layer dimension. The value range is [1, 65535].
    - `H2` indicates the size of the second dimension of the input right matrix, representing the sequence length of the output sequence. The value must be divisible by `rankSize`. The value range is [2, 2147483647].

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp

    def run_npu_matmul_all_to_all(rank, world_size, master_ip, master_port, x1_shape, x2_shape):
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
        output = torch_npu.npu_matmul_all_to_all(
            x1_tensor,
            x2_tensor,
            hcom_info,
            world_size
        )
        print("output: ", output)

    if __name__ == "__main__":
        worksize = 2 # Number of NPUs
        master_ip = '127.0.0.1' # Communication IP address
        master_port = '50001' # Communication port
        x1_shape = [1024, 256] # Input shape of x1
        x2_shape = [256, 3072] # Input shape of x2
        mp.spawn(
            run_npu_matmul_all_to_all,
            args=(worksize, master_ip, master_port, x1_shape, x2_shape),
            nprocs=worksize,
        )
    ```
