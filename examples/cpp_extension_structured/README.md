# 算子适配开发及调用（结构化）

本文档基于C++ extensions方式，torch_npu单算子API进行自定义NPU算子适配开发的完整流程，流程涵盖了算子定义、算子适配、ATen IR注册绑定。本样例重点阐述结构化内核适配方法，该方法适用于aclnn接口与ATen IR语义一致，且适配层逻辑仅需负责output tensor申请的场景。

## 算子适配开发

### 前提条件

在开始之前，请确保您已完成以下环境的安装。

1. 请参考《[CANN 快速安装](https://www.hiascend.com/cann/download)》安装昇腾NPU驱动和CANN软件（包含Toolkit、ops和NNAL包），并配置环境变量。
2. 请参考《[Ascend Extension for PyTorch 软件安装指南](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/zh/installation_guide/installation_description.md)》完成PyTorch框架的安装。

### 适配文件结构

```text
cpp_extension_structured/
├── cpp_extension_structured/
│   └── __init__.py                   # 构建用init文件
├── deprecated.yaml                   # 废弃api配置
├── gen.sh                            # 一键生成脚本：调用 torchnpugen 生成算子适配代码
├── setup.py                          # 项目构建脚本，用于编译生成 whl 包
├── npu_custom.yaml                   # 自定义算子 YAML（含前向/反向 ATen IR 与 aclnn 映射）
├── npu_custom_derivatives.yaml       # 前向/反向绑定配置
├── test_native_functions.yaml        # NPU backend 声明（生成 stub 等时会使用）
├── test/
│   └── test_npu_fast_gelu_custom.py  # 自定义算子测试脚本
└── README.md
```

### 操作步骤

> [!NOTE]
>
> 结构化适配暂未支持前反向绑定，用户可参考[cpp_extension_full/module](../cpp_extension_full/module/README.md)章节通过Python绑定。

1. 算子适配层c++代码目录（csrc）中，通过`npu_custom.yaml`文件完成结构化适配的配置。

    - `func`：PyTorch侧暴露的算子签名（ATen IR格式）。

    - `gen_opapi`：指定输出张量的形状（size）和数据类型（dtype）由哪个输入张量推导（例如self或grad）。

    - `exec`：指定实际调用的底层aclnn接口名称。

    具体示例如下：

    ```yaml
    custom:
    - func: npu_fast_gelu_custom(Tensor self) -> Tensor
      op_api: all_version
      gen_opapi:
        out:
          size: self
          dtype: self
        exec: aclnnFastGelu
    - func: npu_fast_gelu_custom_backward(Tensor grad, Tensor self) -> Tensor
      op_api: all_version
      gen_opapi:
        out:
          size: grad
          dtype: grad
        exec: aclnnFastGeluBackward
    ```

2. 在`cpp_extension_structured`目录下的`__init__.py`文件中，读取so文件。

    ```Python
    import pathlib
    import torch
    # Load the custom operator library
    def _load_opextension_so():
        so_dir = pathlib.Path(__file__).parents[0]
        so_files = list(so_dir.glob('custom_cpp_extension_structured_lib*.so'))
        if not so_files:
            raise FileNotFoundError(f"not find custom_cpp_extension_structured_lib*.so in {so_dir}")
        so_path = str(so_files[0])
        torch.ops.load_library(so_path)
    _load_opextension_so()
    ```

## 调用样例

完成了算子适配开发后，即可实现C++ extensions的方式调用自定义算子。

1. 完成自定义算子工程创建、算子开发及编译部署流程，具体可参考《[CANN Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html)》。

2. 下载示例代码。

    ```bash
    # 下载样例代码
    git clone https://gitcode.com/Ascend/op-plugin
    # 进入代码目录
    cd examples/cpp_extension_structured
    ```

3. 完成算子适配，具体可参考[适配开发](#算子适配开发)。

4. 执行如下命令，完成编译、安装、测试。

    ```bash
    bash build_and_run.sh
    ```

    得到结果如下即为执行成功。

    ```bash
    Ran xx tests in xx s
    OK
    ```
