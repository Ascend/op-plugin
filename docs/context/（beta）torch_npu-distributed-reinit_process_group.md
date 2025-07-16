# （beta）torch_npu.distributed.reinit_process_group

## 函数原型

```
torch_npu.distributed.reinit_process_group(group: Optional[ProcessGroup] = None, rebuild_link: bool = True) -> None
```

## 功能说明

重新构建processgroup集合通信域。

## 参数说明

- **group**(`Optional[ProcessGroup]`) ：默认值为None，使用默认值是reinit default process group。
- **rebuild_link**(`bool`)：默认值为True，当传入参数为True时会将当前的process group hccl建立的通信链接销毁掉，然后进行重建；如果传入参数为False，表示继续使用原有的通信链接。

## 输入说明

要确保是一个有效的device。

## 支持的型号

- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

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
# 重建group
    torch_npu.distributed.reinit_process_group()
    a = torch.ones(2,2,device=f"npu:{rank}")
    dist.all_reduce(a)
def _multiprocess(world_size,f):
    ctx = mp.get_context('spawn')
    ps = []
    for i in range(world_size):
        p = ctx.Process(target=f, args=(i,world_size))
        p.start()
    for p in ps:
        p.join()
if __name__ == '__main__':
    _multiprocess(4, _do_allreduce)
```

