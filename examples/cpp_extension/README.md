# 自定义算子 C++ 扩展开发示例

本示例演示如何使用 Ascend C 实现自定义算子 Kernel，并通过 C++ 扩展（`cpp_extension`）方式集成到 PyTorch，最终在 Python 侧调用。涵盖工程结构、编译、安装、测试全流程。

## 目录结构

```text
├── examples
|   ├── cpp_extension
|   │   ├── csrc
|   │   │   ├── add_custom.asc          # Add算子实现（含 5 种正确启动方式对比）
|   │   │   ├── trig_inplace_custom.asc # 原地三角函数算子实现
|   │   │   └── pybind11.asc            # pybind绑定自定义算子
|   |   ├── op_extension/
|   |   │   ├── __init__.py             # 扩展加载逻辑
|   |   │   └── ops/
|   |   │       └── __init__.py         # Python API 定义及扩展加载逻辑
|   |   ├── test
|   │   |   └── test.py                 # 测试脚本
|   |   ├── setup.py                    # 编译配置
|   |   └── README.md                   # 说明文档
```

## 新增自定义算子

### 1. Kernel 实现

在 `./csrc/` 下创建 `.asc` 文件，基于 Ascend C 实现算子 Kernel。Ascend C 开发参考[昇腾社区文档](https://www.hiascend.com/ascend-c)。

以 `add_custom.asc` 为例，文件包含三部分：

| 组成 | 说明 |
|------|------|
| `KernelAdd` 类 | 算子设备侧实现，包含 `Init` / `Process` / `CopyIn` / `Compute` / `CopyOut` |
| `add_custom` 核函数 | `__global__ __vector__` 修饰的设备入口 |
| `ascendc_add1/2/3/4/5` | 5 种正确的 Host 侧启动方式，对比不同 stream 获取与队列管理策略 |

**5 种启动方式对比**：

| 方式 | 函数 | stream 获取 | 队列管理 | 适用场景 |
|------|------|------------|---------|---------|
| 1 | `ascendc_add1` | `NPUStream` 对象 | `<<<>>>` 内部清 queue | 简单同步场景 |
| 2 | `ascendc_add2` | `stream(true)` | 清 queue 后直接启动 | 等价方式1 |
| 3 | `ascendc_add3` | `stream(false)` | OpCommand 入 queue | **推荐**，保留流水线性能 |
| 4 | `ascendc_add4` | `stream(true)` | 清 queue + OpCommand 入 queue | 语义直观 |
| 5 | `ascendc_add5` | `stream()` | 等待 queue 完成后启动 | 等价方式2 |

> **推荐方式3**：`stream(false)` 配合 `OpCommand::RunOpApiV2`，保留 TaskQueue 流水线性能，与 TorchNPU 内置算子行为一致。

### 2. Python 模块绑定

在 `pybind11.asc` 中使用 pybind11 将 C++ 函数暴露为 Python 接口。Python 侧仅暴露推荐方式3：

```cpp
namespace ascendc_ops {
// 方式3(推荐): stream(false) + OpCommand::RunOpApiV2
at::Tensor ascendc_add3(const at::Tensor &x, const at::Tensor &y);
at::Tensor run_trig_custom(const at::Tensor &x, const at::Tensor &out_sin, const at::Tensor &out_cos);
}

PYBIND11_MODULE(custom_ops_lib, m)
{
    m.def("custom_add", &ascendc_ops::ascendc_add3, "");
    m.def("custom_trig", &ascendc_ops::run_trig_custom, "");
}
```

### 3. Aten IR 实现

算子通过 `at_npu::native::OpCommand::RunOpApiV2` 入队到 TorchNPU 的 TaskQueue，实现异步下发。推荐方式3的实现：

```cpp
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascendc_ops {
at::Tensor ascendc_add3(const at::Tensor &x, const at::Tensor &y)
{
    // stream(false) 返回 ACL stream 但不清 queue
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 8;
    uint32_t totalLength = 1;
    for (uint32_t size : x.sizes()) {
        totalLength *= size;
    }
    // Launch the custom kernel use <<<>>>
    auto acl_call = [=]() -> int {
        add_custom<<<blockDim, nullptr, acl_stream>>>(
            (uint8_t *)(x.mutable_data_ptr()),
            (uint8_t *)(y.mutable_data_ptr()),
            (uint8_t *)(z.mutable_data_ptr()),
            totalLength);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_add", acl_call);
    return z;
}
}  // namespace ascendc_ops
```

上述主要介绍了自定义算子kernel集成的必备流程。完整的5种正确启动方式见`./csrc/add_custom.asc`，Python侧通过`./csrc/pybind11.asc`仅暴露推荐方式3。

最后，通过创建ops路径，定义python接口，通过`module_name.ops.custom_add`可以调用自定义算子。测试样例如下：

```python
import torch
x = torch.randint(low=1, high=100, size=length, device='cpu', dtype=torch.int)
y = torch.randint(low=1, high=100, size=length, device='cpu', dtype=torch.int)

x_npu = x.npu()
y_npu = y.npu()
output = op_extension.ops.custom_add(x_npu, y_npu)
```

## 运行自定义的算子

### 1. 编译 whl 包

      ```bash
      python setup.py bdist_wheel
      ```

`setup.py` 关键逻辑：

| 步骤 | 实现 | 说明 |
|------|------|------|
| 源码收集 | `glob.glob("csrc/*.asc")` | 自动收集所有 `.asc` 文件 |
| 架构识别 | `get_npu_arch()` | 通过 `npu-smi info` 解析芯片型号，映射到 `dav-2201`/`dav-3510` |
| 依赖路径 | `get_dependency_paths()` | 自动收集 torch / torch_npu / Python 的 include 与 lib 路径 |
| ABI 对齐 | `torch._C._GLIBCXX_USE_CXX11_ABI` | 与 PyTorch ABI 保持一致 |
| 编译器 | `bisheng -x asc` | 使用 CANN 提供的 bisheng 编译器 |

编译产物位于 `dist/op_extension-0.1-*.whl`。

可选环境变量：

```bash
USE_NINJA=1 python setup.py bdist_wheel  # 启用 ninja 加速
```

### 2. 安装 whl 包

      ```bash
      cd dist
      pip install *.whl
      ```
  
  3. 运行样例

      ```bash
      cd test
      python test.py
      ``` 

## 常见问题 (FAQ)
 
### 1. 编译时提示 `bisheng command not found`

**问题原因**：系统中未安装 bisheng 编译器或环境变量未正确配置。
**解决方案**：

- 确保已正确安装 CANN 工具包
- 执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 设置环境变量
- 验证 bisheng 编译器是否可用：`bisheng --version`
 
### 2. 运行时提示 `ModuleNotFoundError: No module named 'op_extension'`

**问题原因**：自定义扩展包未正确安装。
**解决方案**：

- 确保已成功编译并安装了 Wheel 包
- 检查安装路径是否在 Python 的搜索路径中
- 尝试使用 `pip install --force-reinstall *.whl` 重新安装

## 注意事项
 
### 1. 数据类型支持

- 当前实现仅支持 `int32` 数据类型
- 如需支持其他数据类型（如 float32、float16），需修改内核实现
- 修改时需注意数据类型的字节大小和对齐要求
 
### 2. 硬件兼容性

- 本示例基于 Ascend NPU 开发，仅支持 Ascend 系列硬件
- 不同型号的 Ascend NPU 可能需要调整编译参数
- 编译时需指定正确的 NPU 架构：`--npu-arch=dav-2201`
 
### 3. 版本兼容性

- 确保 PyTorch、TorchNPU 和 CANN 版本兼容
- 版本不匹配可能导致编译或运行时错误
- 参考 [TorchNPU 文档](https://gitcode.com/ascend/pytorch#%E5%AE%89%E8%A3%85) 获取兼容版本信息
