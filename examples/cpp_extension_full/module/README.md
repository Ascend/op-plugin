# 适配开发及调用（完整样例-module）

基于C++ extensions方式，通过torch_npu来调用单算子API的适配开发过程，其中前反向绑定通过Python类的`forward`和`backward`注册实现。

## 算子适配开发

### 前提条件

在开始之前，请确保您已完成以下环境的安装。

1. 请参考《[CANN 快速安装](https://www.hiascend.com/cann/download)》安装昇腾NPU驱动和CANN软件（包含Toolkit、ops和NNAL包），并配置环境变量。
2. 请参考《[Ascend Extension for PyTorch 软件安装指南](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/zh/installation_guide/installation_description.md)》完成PyTorch框架的安装。

### 适配文件结构

```text
├── build_and_run.sh                // 自定义算子wheel包编译安装并执行用例的脚本
├── csrc                            // 算子适配层c++代码目录
│   └── add_custom.cpp              // 自定义算子正反向适配代码、ATen IR注册以及绑定
├── cpp_extension_full              // 自定义算子包python侧代码
│   ├── ops.py                      // 定义ops调用
│   └── __init__.py                 // python初始化文件
├── setup.py                        // wheel包编译文件
└── test                            // 测试用例目录
    └── test_add_custom.py          // 执行算子用例脚本
```

### 操作步骤

1. 在算子适配层c++代码目录（csrc）中的`add_custom.cpp`完成C++侧算子代码适配、注册自定义算子schema及绑定具体实现。PyTorch提供TORCH_LIBRARY宏来定义新的命名空间，并在该命名空间里注册schema。注意命名空间的名字必须是唯一的。具体示例如下：

    > [!NOTE]
    > 
    > 多卡场景必须在适配代码中加`const c10::OptionalDeviceGuard device_guard(device_of(Tensor))`保障跨device访问。

    ```cpp
        // 为NPU设备注册前向实现
        at::Tensor add_custom_impl_npu(const at::Tensor& self, const at::Tensor& other)
        {
            const c10::OptionalDeviceGuard device_guard(device_of(self));
            // 创建输出内存
            at::Tensor result = at::empty_like(self);

            at::Scalar alpha = 1.0;

            // 调用aclnn接口计算
            EXEC_NPU_CMD_EXT(aclnnAdd, self, other, alpha, result);
            return result;
        }

        // 为NPU设备注册反向实现
        std::tuple<at::Tensor, at::Tensor> add_custom_backward_impl_npu(const at::Tensor& grad)
        {
            const c10::OptionalDeviceGuard device_guard(device_of(grad));
            at::Tensor result = grad; // 创建输出内存

            return {result, result};
        }

        // 为Meta设备注册前向实现
        at::Tensor add_custom_impl_meta(const at::Tensor& self, const at::Tensor& other)
        {
            return at::empty_like(self);
        }

        // 为Meta设备注册反向实现
        std::tuple<at::Tensor, at::Tensor> add_custom_backward_impl_meta(const at::Tensor& self)
        {
            auto result = at::empty_like(self);
            return std::make_tuple(result, result);
        }

        // 为NPU设备注册前反向实现
        // NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
        TORCH_LIBRARY_IMPL(cpp_extension_full, PrivateUse1, m) {
            m.impl("add_custom", &add_custom_impl_npu);
            m.impl("add_custom_backward", &add_custom_backward_impl_npu);
        }

        // 为Meta设备注册前反向实现
        TORCH_LIBRARY_IMPL(cpp_extension_full, Meta, m) {
            m.impl("add_custom", &add_custom_impl_meta);
            m.impl("add_custom_backward", &add_custom_backward_impl_meta);
        }

        TORCH_LIBRARY(cpp_extension_full, m) {
            m.def("add_custom(Tensor self, Tensor other) -> Tensor");
            m.def("add_custom_backward(Tensor self) -> (Tensor, Tensor)");
        }

    ```

2. 在`cpp_extension_full`目录下的`__init__.py`及`_load.py`文件中，添加ops调用及读取so文件，具体示例如下：

    ```Python
    # __init__.py
    __all__ = ['ops', 'add_custom', 'add_custom_backward']
    from .ops import add_custom, add_custom_backward
    import pathlib
    import torch
    # Load the custom operator library
    def _load_opextension_so():
        so_dir = pathlib.Path(__file__).parents[0]
        so_files = list(so_dir.glob('custom_ops_lib*.so'))
        if not so_files:
            raise FileNotFoundError(f"not find custom_ops_lib*.so in {so_dir}")
        atb_so_path = str(so_files[0])
        torch.ops.load_library(atb_so_path)
    _load_opextension_so()

    # ops.py
    import torch
    def add_custom(self, other):
        return torch.ops.cpp_extension_full.add_custom(self, other)
    def add_custom_backward(grad):
        return torch.ops.cpp_extension_full.add_custom_backward(grad)
    ```

3. 在测试脚本`test_add_custom.py`中实现前反向绑定，可通过Python中的`forward`和`backward`定义算子的前向计算与反向梯度计算逻辑。具体示例如下：

    ```Python
    class AddCustomFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, y):
            # 保存梯度计算需要的tensor
            ctx.save_for_backward(x, y)
            # 执行前向计算
            return torch.ops.cpp_extension_full.add_custom(x, y)
        
        @staticmethod
        def backward(ctx, grad_output):
            # 获取保存的tensor
            x, y = ctx.saved_tensors
            return torch.ops.cpp_extension_full.add_custom_backward(grad_output)
    ```

## 调用样例

完成了算子适配开发后，即可实现C++ extensions的方式调用自定义算子。

1. 完成自定义算子工程创建、算子开发及编译部署流程，具体可参考《[CANN Ascend C算子开发指南](https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html)》。
2. 下载示例代码。

    ```bash
    # 下载样例代码
    git clone https://gitcode.com/Ascend/op-plugin
    # 进入代码目录
    cd examples/cpp_extension_full/module
    ```

3. 完成算子适配，具体可参考[算子适配开发](#算子适配开发)。
4. 执行如下命令，完成编译、安装、测试。

    ```bash
    bash build_and_run.sh
    ```

    得到结果如下即为执行成功。
    
    ```bash
    Ran xx tests in xx s
    OK
    ```
