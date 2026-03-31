# torch_npu.npu.NpuGraphOpHandler

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √   |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √   |
| Atlas 推理系列加速卡产品 | √   |

## 功能说明

本接口用于注册自定义算子处理器，使自定义算子支持NPU Graph的动态Shape更新与重放功能。在NPU Graph模式下，用户调用 `g.update()` 传入新的参数时，Ascend Extension for PyTorch框架通过注册的处理器将数据映射到算子输入位置。

核心机制：

1. Capture预处理：定义算子捕获时的输入数据预处理逻辑。
2. Update动态更新：在Graph Replay（回放）阶段，无需重新Capture图结构，即可动态修改算子输入参数（如序列长度、Batch Size等）的机制。具体流程如下：
    1. 用户在Replay前调用`g.update(cpu_update_input=[...])`传入新参数。
    2. 框架遍历Graph中的算子，查找注册的`NpuGraphOpHandler`。
    3. 框架调用Handler的`update_args`方法，传入`dispatch_record` 和`update_input`。
    4. 在`update_args`中，用户直接修改`dispatch_record.args`指定索引的值。

## 定义文件

torch_npu/npu/_npugraph_handlers/npugraph_handler.py

## 函数原型

### NpuGraphOpHandler基类

```python
class NpuGraphOpHandler:
    @classmethod
    def prepare_capture(cls, func, args, kwargs): ...
    @classmethod
    def postprocess_result(cls, result, kwargs): ...
    @classmethod
    def update_args(cls, dispatch_record, update_input): ...
    @classmethod
    def record_wrap_kwarg(cls, key, value, tensor_param_names): ...
```

### register_npu_graph_handler装饰器

```python
def register_npu_graph_handler(op_names: str | list[str]): ...
```

## 参数说明

### NpuGraphOpHandler基类

#### 1. prepare_capture

- **func**（`Callable`）：原始算子函数对象。
- **args**（`Tuple`）：位置参数元组。
- **kwargs**（`Dict`）：关键字参数字典。

#### 2. postprocess_result

- **result**（`Any`）：算子执行的原始结果。
- **kwargs**（`Dict`）：关键字参数字典。

#### 3. update_args

- **dispatch_record**（`DispatchRecord`）：包含算子运行时信息的记录对象，可通过 `dispatch_record.args` 修改参数。
- **update_input**（`Dict`）：用户传入的更新参数字典。

#### 4. record_wrap_kwarg

- **key**（`str`）：关键字参数的键名。
- **value**（`Any`）：关键字参数的值。
- **tensor_param_names**（`List[str]`）：张量参数名称列表。

> [!CAUTION]<br>
> 所有方法必须声明为 `classmethod`，禁止使用 `self` 存储状态（无状态设计）。

### register_npu_graph_handler装饰器

**op_names**（`str` 或 `list[str]`）：算子名称，对应 `func.__name__`。建议同时注册 `op_name` 和 `op_name.default`。

## 返回值说明

### NpuGraphOpHandler基类

#### 1. prepare_capture

返回处理后的 `func, args, kwargs`。用于output预分配或算子变体替换。默认透传。源码中常用于切换`.default`到`.out`并预分配workspace。

#### 2. postprocess_result

返回处理后的结果。用于调整返回值格式。默认返回result。源码中常用于将C++ TensorList转为 Python list。

#### 3. update_args

无返回值。用于修改位置参数以响应动态更新。默认空实现。**核心更新逻辑在此实现**。

#### 4. record_wrap_kwarg

返回处理后的值。用于自定义kwargs存储方式。默认处理Tensor弱引用。

### register_npu_graph_handler装饰器

返回被装饰的原始类。 

## 调用示例

本示例展示如何自定义一个Handler，同时实现输出预分配（`prepare_capture`）、返回值格式调整（`postprocess_result`）、动态参数更新（`update_args`）以及kwargs自定义存储（`record_wrap_kwarg`）。

```python
import torch
import torch_npu
from torch_npu.npu import (
    NpuGraphOpHandler,
    register_npu_graph_handle
)

@register_npu_graph_handler(["my_custom_op.default"])
class ComprehensiveOpHandler(NpuGraphOpHandler):
    # 定义参数映射：第 3 个参数 (index 2) 对应 update_input 中的 "seq_len"
    _OP_ARG_SPECS = {
        "my_custom_op.default": (2, "seq_len"),
    }

    @classmethod
    def prepare_capture(cls, func, args, kwargs):
        # 1. 切换到 .out 算子并预分配 output
        func_out = torch.ops.my_namespace.my_custom_op.out
        if "out" not in kwargs:
            # 预分配逻辑，例如根据输入形状分配
            kwargs["out"] = [torch.empty_like(args[0]) for _ in range(2)]
        return func_out, args, kwargs

    @classmethod
    def postprocess_result(cls, result, kwargs):
        # 2. 调整返回值格式，将底层 TensorList 转为 Python list
        return kwargs["out"]

    @classmethod
    def update_args(cls, dispatch_record, update_input):
        # 3. 动态更新参数逻辑
        spec = cls._OP_ARG_SPECS.get(dispatch_record.op_cache_entry.__name__)
        if spec:
            arg_index, key = spec
            # 检查 key 是否存在并更新 args
            if key in update_input and len(dispatch_record.args) >= (arg_index + 1):
                dispatch_record.args[arg_index] = update_input[key]

    @classmethod
    def record_wrap_kwarg(cls, key, value, tensor_param_names):
        # 4. 自定义 kwargs 存储方式，例如对 Tensor 进行弱引用处理
        if key in tensor_param_names and isinstance(value, torch.Tensor):
            return torch._C._weak_ref(value)
        return value

# 使用流程
g = torch.npu.NPUGraph()
with torch.npu.graph(g, auto_dispatch_capture=True):
    # Capture 阶段：调用算子，触发 prepare_capture 和 record_wrap_kwarg
    out_list = torch.ops.my_namespace.my_custom_op(x, weight, seq_len)

# Replay 阶段：传入新参数，触发 update_args，最后触发 postprocess_result
g.update(cpu_update_input=[{"seq_len": new_seq_len}])
g.replay()
```
