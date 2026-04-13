# 适配开发及调用

## 目录结构介绍

```text
├── examples
|   ├── cpp_extension
|   │   ├── csrc
|   │   │   ├── add_custom.asc          # Add算子实现
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

本示例以Add算子为例，展示了如何在 PyTorch 中使用Ascend C扩展自定义算子Kernel实现，并通过Python的接口调用实现的算子。

### kernel实现

  本小节主要介绍如何实现Kernel算子，本样例基于Ascend C进行开发算子。如何使用Ascend C实现算子kernel，可以参考昇腾社区文档[昇腾Ascend C](https://www.hiascend.com/ascend-c)。

  在./csrc/ 目录下创建一个名为`add_custom.asc`的文件，这是我们自定义加法Kernel的实现文件。样例中实现了一个`run_ascendc_add`的核函数。

### 将Kernel实现与集成

  本小节主要介绍如何封装实现的kernel算子，以及绑定为Python的接口。
  代码实现在./csrc/ 目录下。

#### 封装Python模块

在`pybind11.asc`文件中使用了pybind11库来将C++代码封装成Python模块，在Python侧可以通过`import`方式进行调用。例如：
  
  ```c++
  PYBIND11_MODULE(custom_ops, m)
  {
      m.def("custom_add", &ascendc_ops::run_ascendc_add, "");
  }
  ```

  通过此绑定，python侧可通过`op_extension.ops.custom_add`调用自定义的API。

#### Aten IR实现

根据Aten IR定义适配算子。
TORCH_NPU的算子下发和执行是异步的，通过TASKQUEUE实现，
样例中，我们通过`at_npu::native::OpCommand::RunOpApiV2`方法，将算子执行入队到TORCH_NPU的TASKQUEUE。样例如下：

  ```c++
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
namespace ascendc_ops {
at::Tensor run_ascendc_add(const at::Tensor &x, const at::Tensor &y)
{
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 8;
    uint32_t totalLength = 1;
    for (uint32_t size : x.sizes()) {
        totalLength *= size;
    }
    // Launch the custom kernel use <<<>>>
    auto acl_call = [=]() -> int{
        add_custom<<<blockDim, nullptr, acl_stream>>>((uint8_t *)(x.mutable_data_ptr()), (uint8_t *)(y.mutable_data_ptr()),
                                                    (uint8_t *)(z.mutable_data_ptr()), totalLength);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_add", acl_call);

    return z;
}

}  // namespace ascendc_ops
  ```

上述主要介绍了一个自定义算子kernel如何集成必备流程。

最后，通过创建ops路径，定义python接口，通过`module_name.ops.custom_add`可以调用自定义算子。测试样例如下：

  ```c++
import torch
x = torch.randint(low=1, high=100, size=length, device='cpu', dtype=torch.int)
y = torch.randint(low=1, high=100, size=length, device='cpu', dtype=torch.int)

x_npu = x.npu()
y_npu = y.npu()
output = op_extension.ops.custom_add(x_npu, y_npu)
  ```

## 运行自定义的算子

  运行依赖torch、torch_npu和CANN。具体安装步骤参考[torch_npu文档](https://gitcode.com/ascend/pytorch#%E5%AE%89%E8%A3%85)
  运行流程：

  1. 运行setup脚本，编译生成whl包。
    ```bash
    python setup.py bdist_wheel
    ```
  我们的编译工程通过setuptools已为用户封装好如何编译算子kernel和集成到pytorch。

  2. 安装whl包
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

- 确保 PyTorch、torch_npu 和 CANN 版本兼容
- 版本不匹配可能导致编译或运行时错误
- 参考 [torch_npu 文档](https://gitcode.com/ascend/pytorch#%E5%AE%89%E8%A3%85) 获取兼容版本信息
