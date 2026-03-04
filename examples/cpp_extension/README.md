## 目录结构介绍
```
├── examples
|   ├── cpp_extension
|   │   ├── csrc
|   │   │   └── add_custom.asc      # Ascend C 内核实现
|   |   ├── op_extension/
|   |   │   ├── __init__.py         # 扩展加载逻辑
|   |   │   └── ops.py              # Python API 定义
|   |   ├── test
|   │   |   └── test.py             # 测试脚本
|   |   ├── setup.py                # 编译配置
|   |   └── README.md               # 说明文档
```
## 新增自定义算子
本示例展示了如何在 PyTorch 中使用Ascend C扩展自定义算子Kernel实现，并通过Pytorch的API调用实现的算子。
### kernel实现
  本小节主要介绍如何实现Kernel算子，本样例基于Ascend C进行开发算子。如何使用Ascend C实现算子kernel，可以参考昇腾社区文档[昇腾Ascend C](https://www.hiascend.com/ascend-c)。


  在./cpp_extension/csrc/ 目录下创建一个名为 `add_custom.asc` 的文件，这是我们自定义加法Kernel的实现文件。样例中实现了一个 `add_custom` 的核函数。

### 将Kernel实现与Pytorch集成
  本小节主要介绍如何封装实现的kernel算子，以及注册为pythorch的API。
  代码实现在./cpp_extension/csrc/ 目录下。

#### Aten IR定义
pytorch通过`TORCH_LIBRARY`宏提供了一个简单方法，将C++算子实现绑定到python。python侧可以通过`torch.ops.namespace.op_name`方式进行调用。这种方式包括Aten IR的定义和注册。我们通过`TORCH_LIBRARY`实现注册，如果相同namespace在不同文件中有注册，需要使用`TORCH_LIBRARY_FRAGMENT`。
例如将自定义`ascendc_add`函数注册在`npu`命名空间下：
  ```c++
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("ascendc_add(Tensor x, Tensor y) -> Tensor");
}
  ```
  通过此注册，python侧可通过`torch.ops.npu.ascendc_add`调用自定义的API。

#### Aten IR实现

根据Aten IR定义适配算子。
TORCH_NPU的算子下发和执行是异步的，通过TASKQUEUE实现，
样例中，我们通过`at_npu::native::OpCommand::RunOpApiV2`方法，将算子执行入队到TORCH_NPU的TASKQUEUE。样例如下：

  ```c++
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
namespace ascendc_ops {
at::Tensor ascendc_add(const at::Tensor &x, const at::Tensor &y)
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
#### Aten IR注册
通过pytorch提供的`TORCH_LIBRARY_IMPL`注册算子实现，运行在NPU设备上需要注册在`PrivateUse1`这个dispatchkey上。
样例如下：
  ```c++
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("ascendc_add", TORCH_FN(ascendc_ops::ascendc_add));
}
  ```

上述主要介绍了一个自定义算子kernel如何集成在pytorch必备流程。

最后，通过创建ops.py文件，定义python接口，通过`module_name.ops.custom_add`可以调用自定义算子。setup.py文件中定义了`module_name`为`op_extension`，样例如下：
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
 
### 3. 运行时提示 `RuntimeError: not find custom_ops_lib*.so`
**问题原因**：编译生成的共享库文件未找到。
**解决方案**：
- 检查编译过程是否成功生成了 `.so` 文件
- 确保安装路径正确，库文件在预期位置
- 检查 `__init__.py` 中的库文件加载逻辑

## 注意事项
 
### 1. 数据类型支持
- 当前实现仅支持 `int32` 数据类型
- 如需支持其他数据类型（如 float32、float16），需修改内核实现
- 修改时需注意数据类型的字节大小和对齐要求
 
### 2. 硬件兼容性
- 本示例基于 Ascend NPU 开发，仅支持 Ascend 系列硬件
- 不同型号的 Ascend NPU 可能需要调整编译参数
- 编译时需指定正确的 NPU 架构：`--npu-arch=dav-2201`
 
### 4. 版本兼容性
- 确保 PyTorch、torch_npu 和 CANN 版本兼容
- 版本不匹配可能导致编译或运行时错误
- 参考 [torch_npu 文档](https://gitcode.com/ascend/pytorch#%E5%AE%89%E8%A3%85) 获取兼容版本信息
