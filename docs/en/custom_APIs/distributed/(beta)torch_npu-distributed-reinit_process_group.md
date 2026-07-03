# (beta) torch_npu.distributed.reinit_process_group

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

Rebuilds the `ProcessGroup` for collective communication.

## Prototype

```python
torch_npu.distributed.reinit_process_group(group: Optional[ProcessGroup] = None, rebuild_link: bool = True) -> None
```

## Parameters

- **`group`** (`Optional[ProcessGroup]`): Optional. Communication group. The default value is `None`.
- **`rebuild_link`** (`bool`): Optional. When set to `True`, the framework destroys and rebuilds the communication links created by the current `ProcessGroupHCCL` instance. When set to `False`, execution reuses the existing communication links. The default value is `True`.

## Return Values

None

## Constraints

The specified input device must be valid.

## Example

```python
import os
import torch
import torch.distributed as dist
import multiprocessing as mp
import torch_npu
def _do_allreduce(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl',
                            world_size=world_size,
                            rank=rank)
# Rebuild a group.
    torch_npu.distributed.reinit_process_group()
    a = torch.ones(2,2,device=f"npu:{rank}")
    dist.all_reduce(a)
def _multiprocess(world_size,f):
    ctx = mp.get_context('spawn')
    ps = []
    for i in range(world_size):
        p = ctx.Process(target=f, args=(i,world_size))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
if __name__ == '__main__':
    _multiprocess(4, _do_allreduce)
```
