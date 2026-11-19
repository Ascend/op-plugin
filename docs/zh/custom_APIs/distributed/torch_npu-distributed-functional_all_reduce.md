# （beta）torch_npu.distributed.functional_all_reduce

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

在指定通信组内，对各rank输入Tensor的对应元素执行规约，并在每个rank返回规约结果。

- 自动分配输出Tensor，底层通信使用独立的发送缓冲和接收缓冲。
- 面向CCU_MS前向通信场景。通信算子在CCU（Collective Communication Unit，集合通信加速单元）上展开，采用MS（Memory Slice，内存切片）模式，通过片上缓冲中转通信数据。
- 支持Eager模式和NPUGraph_EX编译模式。
- 返回结果支持自动等待，仅`"sum"`支持自动求导。

## 函数原型

```python
torch_npu.distributed.functional_all_reduce(
    self,
    reduceOp,
    group,
    tag="",
    *,
    premul_factor=None,
) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：必选参数，NPU上的稠密输入Tensor。各rank的输入须满足同一次AllReduce的shape和dtype一致性要求。支持的dtype见约束说明。
- **reduceOp** (`str`)：必选参数，规约类型，大小写不敏感。
    - `"sum"`：对对应元素求和。
    - `"avg"`：对对应元素求平均值。整数结果采用向零截断。
    - `"min"`：取对应元素的最小值。
    - `"max"`：取对应元素的最大值。
    - `"premul_sum"`：本rank输入先乘以`premul_factor`，再对各rank的预乘结果求和。
    - 传入ReduceOp对象时，抛出`TypeError`。
- **group**：必选参数，指定已创建的HCCL通信组。支持ProcessGroup、通信组名称、一维DeviceMesh，以及通过`(DeviceMesh, int)`指定Mesh维度的形式。使用默认进程组时，显式传入`torch.distributed.group.WORLD`。传入`None`时抛出`ValueError`。
- **tag** (`str`)：可选参数，默认值为`""`。在兼容的rank列表调用形式中，与rank列表共同用于查找通信组，该调用形式已在配套PyTorch中弃用。传入ProcessGroup、通信组名称或DeviceMesh时，使用其关联的既有通信组。
- **premul_factor** (`int`、`float`)：可选关键字参数，默认值为`None`，表示`"premul_sum"`的预乘因子。
    - `"premul_sum"`必须提供该参数，缺失时抛出`RuntimeError`。
    - `"sum"`、`"avg"`、`"min"`、`"max"`应省略该参数，提供数值乘数时抛出`RuntimeError`。
    - 传入bool、Tensor或其他类型时，抛出`TypeError`。

## 返回值说明

`Tensor`

返回规约结果，其shape、dtype和device与`self`一致，输出存储独立分配。

> [!NOTE]
>
> 通信采用异步下发，函数返回时通信不一定已完成。返回Tensor用于后续计算时，functional通信机制会建立必要的等待依赖。

## 约束说明

**数据类型与支持的规约操作**

| 输入dtype | 支持的前向规约操作 |
| --- | --- |
| `torch.float16`、`torch.float32`、`torch.bfloat16` | `"sum"`、`"avg"`、`"min"`、`"max"`、`"premul_sum"` |
| `torch.int16`、`torch.int32`、`torch.uint8`、`torch.bool` | `"sum"`、`"avg"`、`"min"`、`"max"` |
| `torch.complex64` | `"sum"`、`"avg"` |

- `bool`、`uint8`使用INT32通信表示，`complex64`使用FP32实数视图，返回时恢复对应类型。
- 复数输入为未解析的共轭视图时，须先调用`resolve_conj()`处理。
- 仅`"sum"`支持反向计算。其他规约在请求反向时抛出`RuntimeError`。`"sum"`反向通过标准functional通信路径汇总梯度。

**通信配置**

- 使用HCCL通信组，并在初始化通信器前设置`HCCL_OP_EXPANSION_MODE=CCU_MS`。
- 进程组的`hccl_op_expansion_mode`配置须保持默认值，`HCCL_DETERMINISTIC=strict`会被拒绝。
- 运行环境须满足CCU_MS的硬件、拓扑和资源要求。HCCL可能根据实际运行条件选择回退路径。
- 通信组须提供未包装的HCCL backend。`TORCH_DISTRIBUTED_DEBUG=DETAIL`创建的ProcessGroupWrapper会被拒绝。
- 空Tensor和单rank调用仍遵循设备及配置检查，这些场景可能不产生实际通信任务。

**输入使用责任**

普通规约使用满足通信要求的输入作为发送数据。必要的格式、连续性和dtype转换可能分配临时存储，`"premul_sum"`会单独分配预乘结果。

调用者须保证输入及其别名在通信完成前保持有效、稳定。需要隔离原始数据或提前修改原始输入时，在调用处传入`self.clone()`。启用NPU OpHook时，回调可能接触调用者存储，回调应保持只读，或由调用者预先复制输入。

## 调用示例

在Ascend 950DT单机8个rank环境中，创建以下文件test.py并保存。

```python
import os

import torch
import torch.distributed as dist
import torch_npu
from torch_npu.distributed import functional_all_reduce


def reduce_sum(value):
    return functional_all_reduce(
        value.clone(), "sum", dist.group.WORLD
    )


local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)
dist.init_process_group(backend="hccl")

try:
    rank = dist.get_rank()
    x = torch.full(
        (4,),
        float(rank + 1),
        device=f"npu:{local_rank}",
        requires_grad=True,
    )

    y = reduce_sum(x)
    y.sum().backward()
    print(rank, "sum:", y.detach().cpu())
    print(rank, "gradient:", x.grad.cpu())

    z = functional_all_reduce(
        x.detach().clone(),
        "premul_sum",
        dist.group.WORLD,
        premul_factor=0.5,
    )
    print(rank, "premul_sum:", z.cpu())

    compiled = torch.compile(
        reduce_sum, backend="npugraph_ex", fullgraph=True
    )
    print(rank, "compiled sum:", compiled(x.detach()).cpu())
finally:
    dist.destroy_process_group()
```

执行如下命令。

```bash
HCCL_OP_EXPANSION_MODE=CCU_MS TORCH_DISTRIBUTED_DEBUG=OFF \
torchrun --standalone --nproc-per-node=8 test.py
```

在8个rank条件下，各rank的`"sum"`和编译后的`"sum"`输出元素应为`36`，`"premul_sum"`输出元素应为`18`，输入梯度元素应为`8`。该示例展示接口使用方式，运行结果以实际设备执行为准。
