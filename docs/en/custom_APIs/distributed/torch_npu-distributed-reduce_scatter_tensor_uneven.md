# (beta) torch_npu.distributed.reduce_scatter_tensor_uneven

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 training products/Atlas 800I A2 inference products</term>           |    √     |
|<term>Atlas inference products</term>                                      |    √     |

## Function

Extends the native [torch.distributed.reduce_scatter_tensor](https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter_tensor) API by supporting zero-copy and uneven tensor splitting in <code>torch_npu.distributed.reduce_scatter_tensor_uneven</code>.

## Prototype

```python
torch_npu.distributed.reduce_scatter_tensor_uneven(output, input, input_split_sizes =None, op=dist.ReduceOp.SUM, group=None, async_op=False) -> torch.distributed.distributed_c10d.Work
```

## Parameters

- **`output`** (`Tensor`): Required. Output tensor used to receive the computation result.
- **`input`** (`Tensor`): Required. Input tensor providing the source data. The shape of `input` corresponds to the concatenation of `output` tensors across all ranks.
- **`input_split_sizes`** (`List[int]`): Optional. Specifies the split sizes along dimension 0 of the `input` tensor. The default value is `None`. The number of elements must match the size of the `group`, and the sum of elements must equal the size of dimension 0 of `input`.
    - If the sum of `input_split_sizes` does not match the size of dimension 0 of `input`, a `RuntimeError` is raised: "Split sizes doesn't match total dim 0 size."
    - If the number of elements in `input_split_sizes` does not match the size of `group`, a `RuntimeError` is raised: "Number of tensor splits not equal to group size."
- **`op`** (`torch._C._distributed_c10d.ReduceOp.ReduceOpType`): Optional. Reduction operator used to control the computation behavior. The default value is `dist.ReduceOp.SUM`.
- **`group`** (`torch.distributed.distributed_c10d.ProcessGroup`): Optional. The process group for distributed communication. The default value is `None`.
- **`async_op`** (`bool`): Optional. Specifies whether to execute the operation asynchronously. The default value is `False`.

## Return Values

A work handle used to track the asynchronous operation. The final result is written to `output`.
No constraints are imposed on the shape of `output`.

## Constraints

- This API can be used only in single-server scenarios.

- The sum of `input_split_sizes` must equal the size of dimension 0 of `input`. The number of elements in `input_split_sizes` must match the size of the `group`.

## Example

Create the test.py file and save it.

```python
import os
import torch
import torch_npu
import torch.distributed as dist
dist.init_process_group(backend="hccl")
rank = int(os.getenv('LOCAL_RANK'))
torch.npu.set_device(rank)
input_split_sizes = [2, 3]
input_tensor = torch.ones(sum(input_split_sizes), dtype=torch.int32).npu()
output_tensor = torch.zeros(input_split_sizes[rank], dtype=torch.int32).npu()
torch_npu.distributed.reduce_scatter_tensor_uneven(
    output_tensor,
    input_tensor,
    input_split_sizes=input_split_sizes,
    async_op=False
)
```

Run the following command:

```bash
torchrun --nproc-per-node=2 test.py
```
