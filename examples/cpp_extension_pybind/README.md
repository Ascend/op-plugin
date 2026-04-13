# 适配开发及调用（pybing）

本文档基于C++ extensions方式，使用torch_npu单算子API进行适配开发的完整流程。流程涵盖了算子定义、算子适配、ATen IR注册绑定，最终实现调用自定义算子。区别于常见的TORCH_LIBRARY方式，本用例采用PYBIND进行绑定注册，以获得更灵活的输入类型支持。

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
├── cpp_extension_pybind            // 自定义算子包python侧代码
│   ├── __init__.py                 // python初始化文件
│   └── ops                         // 定义ops调用
│       └── __init__.py             // python初始化文件
├── setup.py                        // wheel包编译文件
└── test                            // 测试用例目录
    └── test_add_custom.py          // 执行eager模式下算子用例脚本
```

### 操作步骤

1. 在算子适配层c++代码目录（csrc）中，通过`add_custom.cpp`文件完成C++侧的算子代码适配、注册自定义算子schema及具体实现的绑定。具体示例如下：

    > [!NOTE]
    > 
    > 以下是基于单卡场景提供的示例代码。在此场景下，`const c10::OptionalDeviceGuard device_guard(device_of(self));`语句是可选的。在多卡场景中，则必须在适配代码中加入此语句。   

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

    PYBIND11_MODULE(custom_ops_lib, m)
    {
        m.def("add_custom", &add_custom_impl_npu, "");
    }
    ```

2. 在`cpp_extension_base`目录下的`__init__.py`及`ops.py`文件中，添加ops调用及so读取。

    ```Python
    # __init__.py
    __all__ = ['ops', 'add_custom']
    from .ops import add_custom

    # ops/__init__.py
    __all__ = ["add_custom"]
    from cpp_extension_pybind.custom_ops_lib import add_custom
    ```

## 调用样例

完成了算子适配开发后，即可实现C++ extensions的方式调用自定义算子。

1. 完成自定义算子工程创建、算子开发及编译部署流程，具体可参考《[CANN Ascend C算子开发指南](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0002.html)》。

2. 下载示例代码。

    ```bash
    # 下载样例代码
    git clone https://gitcode.com/Ascend/op-plugin
    # 进入代码目录
    cd examples/cpp_extension_pybind
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
    