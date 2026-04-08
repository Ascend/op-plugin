# 适配开发及调用（基础样例）

本文档基于C++ extensions方式，使用torch_npu单算子API进行适配开发的完整流程。流程涵盖了算子定义、算子适配、ATen IR注册绑定，最终实现调用自定义算子。

## 算子适配开发

### 前提条件

在开始之前，请确保您已完成以下环境的安装。
1. 完成CANN软件的安装具体请参见《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html)》（商用版）或《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html)》（社区版）。
2. 完成PyTorch框架的安装具体请参见《[Ascend Extension for PyTorch 软件安装指南](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/zh/installation_guide/installation_description.md)》。

### 适配文件结构
```text
├── build_and_run.sh                // 自定义算子wheel包编译安装并执行用例的脚本
├── csrc                            // 算子适配层c++代码目录
│   └── add_custom.cpp              // 自定义算子正反向适配代码、aten ir注册以及绑定
├── cpp_extension_base              // 自定义算子包python侧代码
│   ├── ops.py                      // 定义ops调用
│   └── __init__.py                 // python初始化文件
├── setup.py                        // wheel包编译文件
└── test                            // 测试用例目录
    └── test_add_custom.py          // 执行eager模式下算子用例脚本
```

### 操作步骤



1. 在算子适配层c++代码目录（csrc）中，通过`add_custom.cpp`文件完成C++侧的算子代码适配、注册自定义算子schema及具体实现的绑定。PyTorch提供的TORCH_LIBRARY宏用于定义唯一命名空间（该命名空间的名称必须保证全局唯一性），并在该命名空间中注册算子schema。具体示例如下：

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

    // 为NPU设备注册前反向实现
    // NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
    TORCH_LIBRARY_IMPL(cpp_extension_base, PrivateUse1, m) {
        m.impl("add_custom", &add_custom_impl_npu);
    }

    TORCH_LIBRARY(cpp_extension_base, m) {
        m.def("add_custom(Tensor self, Tensor other) -> Tensor");
    }
    ```

2. 在`cpp_extension_base`目录下的`__init__.py`及`ops.py`文件中，添加ops调用及读取so文件，具体示例如下：

    ```Python
    # __init__.py
    __all__ = ['ops', 'add_custom']
    from .ops import add_custom
    import pathlib
    import torch
    def _load_opextension_so():
        so_dir = pathlib.Path(__file__).parents[0]
        so_files = list(so_dir.glob('custom_ops_lib*.so'))

        if not so_files:
            raise FileNotFoundError(f"not find custom_ops_lib*.so in {so_dir}")

        so_path = str(so_files[0])
        torch.ops.load_library(so_path)
    _load_opextension_so()

    # ops.py
    import torch
    def add_custom(self, other):
        return torch.ops.cpp_extension_base.add_custom(self, other)
    ```

## 调用样例

完成了算子适配开发后，即可通过C++ extensions的方式调用自定义算子。

1. 完成自定义算子工程创建、算子开发及编译部署流程，具体可参考《[CANN Ascend C算子开发指南](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0002.html)》。
2. 下载示例代码。

    ```bash
    # 下载样例代码
    git clone https://gitcode.com/Ascend/op-plugin
    # 进入代码目录
    cd examples/cpp_extension_base
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