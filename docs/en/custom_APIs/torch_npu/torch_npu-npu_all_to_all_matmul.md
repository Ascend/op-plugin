# torch_npu.npu_all_to_all_matmul

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->

## Function

- **Function**: fuses the AlltoAll communication, Permute (to ensure contiguous addresses after communication), and Matmul computation. Communication is performed before computation.

- **Formulas**:

    Assume that the shape of the input x1 is (BS, H), the shape of x2 is (H\*rankSize, N), and rankSize is the number of NPU devices.

    $$
    commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
    permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
    output = permutedOut @ x2 + bias \\
    $$

## Prototype

```python
torch_npu.npu_all_to_all_matmul(x1, x2, hcom, world_size, bias=None, all2all_axes=None, all2all_out_flag=True, comm_mode=None) -> (Tensor, Tensor)
```

## Parameters

- **`x1`** (`Tensor`): Required. Left matrix input of the fused operator and the left operand of MatMul computation, $x1$ in the formulas. This parameter must be a 2D tensor, with shape `(BS, H)`. The data type can be `torch.bfloat16` or `torch.float16`. The data layout is ND. Non-contiguous tensors are not supported. Empty tensors where the first dimension size is `0` are supported.
- **`x2`** (`Tensor`): Required. Right matrix input of the fused operator and the right operand of MatMul computation, $x2$ in the formulas. This parameter must be a 2D tensor, with shape `(H*rankSize, N)`. The data type must be identical to that of `x1`. The data layout can be ND. Transposed non-contiguous tensors are supported.
- **`hcom`** (`str`): Required. Communication domain identifier string on the host side, representing the communication domain name obtained using the `get_hccl_comm_name` API.
- **`world_size`** (`int`): Required. Total number of ranks within the communication domain, $rankSize$ in the formulas. Valid values are `2`, `4`, `8`, or `16`.
- **`bias`** (`Tensor`): Optional. Accumulative bias added after the matrix multiplication computation, $bias$ in the formulas. When the data type of `x1` and `x2` is `torch.float16`, the data type of `bias` is `torch.float16`. When the data type of `x1` and `x2` is `torch.bfloat16`, the data type of `bias` is `torch.float32`. This parameter must be a 1D tensor, with shape `(N,)`. The data layout can be ND.
- **`all2all_axes`** (`List[int]`): Optional. Data exchange axis configuration for the AlltoAll and Permute operations. This parameter can be empty or be set to `[-2, -1]`, indicating that the input `x1` is converted from `(BS, H)` to `(BS/rankSize, H*rankSize)`.
- **`all2all_out_flag`** (`bool`): Optional. Specifies whether to output the result after `AlltoAll` and `Permute` operations. The default value is `True`.
- **`comm_mode`** (`str`): **Optional**. Communication engine mode. The default value is `None`.

    <!-- npu="A3" id3 -->
    - <term>Atlas A3 products</term>: the supported values are `None` and `ai_cpu`. When `None` is passed, `ai_cpu` communication is used.
    <!-- end id3 -->
    <!-- npu="950" id4 -->
    - <term>Ascend 950PR&950DT products</term>: the supported values are `None`, `ai_cpu`, and `ccu`. When `None` is passed, `ai_cpu` communication is used.
    <!-- end id4 -->

## Return Values

- **`y`** (`Tensor`): Computation output tensor representing the final computation result. This parameter must be a 2D tensor, with shape `(BS/rankSize, N)`. The data type is identical to that of `x1` or `x2`. The data layout can be ND. Non-contiguous tensors are not supported.
- **`all2all_out`** (`Tensor`): The output tensor after the `AlltoAll` and `Permute` operations, $permutedOut$ in the formulas. When `all2all_out_flag` is `True`, the actual tensor is returned. The data type must be identical to `x1` or `x2`. The shape must be 2D with shape `(BS/rankSize, H*rankSize)`. The data layout can be ND. Non-contiguous tensors are not supported.

## Constraints

- This API can be used in training and inference scenarios.
- This API supports single-operator mode.

<!-- npu="950" id5 -->
- On <term>Ascend 950PR&950DT products</term>, this API supports graph mode calls.
<!-- end id5 -->
- All input parameters except `x1` do not support empty tensors.
- The communication domain name `hcom` must not be an empty string. The length range is [1, 127].
- Variables used in the shapes of the parameters are as follows:
    - BS: size of the first dimension of the input left matrix, indicating the number of input sequences. The value range is [0, 2147483647], and it must be divisible by rankSize.
    - H: size of the second dimension of the input left matrix, indicating the hidden layer dimension. The value range is limited by the number of NPU devices, and the value range of H\*rankSize is [2, 65535].
    - H\*rankSize: size of the first dimension of the input right matrix, indicating the size of the hidden layer dimension of the input left matrix after AlltoAll communication. The value range is [2, 65535].
    - N: size of the second dimension of the input right matrix, indicating the length of the output sequence. The value range is [1, 2147483647].

## Example

- Single-operator call

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import numpy as np

    def run_all_to_all_matmul(rank, world_size, master_ip, master_port, x1_shape, x2_shape):
        torch_npu.npu.set_device(rank)
        init_method = "tcp://" + master_ip + ":" + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > "2.0.1":
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
        x1_tensor = torch.randn(x1_shape, dtype=torch.float16).npu()
        x2_tensor = torch.randn(x2_shape, dtype=torch.float16).npu()
        output, all2all_out = torch_npu.npu_all_to_all_matmul(
            x1_tensor,
            x2_tensor,
            hcom_info,
            world_size
        )
        print("output: ", output)
        print("all2all_out: ", all2all_out)


    if __name__ == "__main__":
        worksize = 2
        master_ip = "127.0.0.1"
        master_port = "50001"
        x1_shape = [1024, 256]
        x2_shape =  [512, 3072]

        mp.spawn(
            run_all_to_all_matmul,
            args=(worksize, master_ip, master_port, x1_shape, x2_shape),
            nprocs=worksize,
        )
    ```

<!-- npu="950" id6 -->
- Graph mode call: Only <term>Ascend 950PR&950DT products</term> are supported.

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import numpy as np
    config = CompilerConfig()
    config.debug.graph_dump.type = 'pbtxt'
    # npu_backend = tng.get_npu_backend(compiler_config=config)
    npu_backend = tng.get_npu_backend(compiler_config=None)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, hcom, world_size):
            return torch_npu.npu_all_to_all_matmul(x1, x2, hcom, world_size)

    def run_all_to_all_matmul_graph(rank, world_size, master_ip, master_port, x1_shape, x2_shape):
        # get hcom_info
        torch_npu.npu.set_device(rank)
        init_method = "tcp://" + master_ip + ":" + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > "2.0.1":
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
        # generate input tensor
        x1_tensor = torch.randn(x1_shape, dtype=torch.float16).npu()
        x2_tensor = torch.randn(x2_shape, dtype=torch.float16).npu()
        cpu_model = Model()
        # dynamic = False -> static graph; dynamic = True -> dynamic graph
        model = torch.compile(cpu_model, backend=npu_backend, dynamic=False, fullgraph=True)
        # model = torch.compile(cpu_model, backend=npu_backend, dynamic=False)
        output = model(
            x1_tensor,
            x2_tensor,
            hcom_info,
            world_size
        )
        print("print output: ", output)
    if __name__ == "__main__":
        world_size = 2
        master_ip = "127.0.0.1"
        master_port = "50001"
        x1_shape = [88, 128]
        x2_shape = [256, 256]
        mp.spawn(
            run_all_to_all_matmul_graph,
            args=(world_size, master_ip, master_port, x1_shape, x2_shape),
            nprocs=world_size,
        )
    ```
<!-- end id6 -->
