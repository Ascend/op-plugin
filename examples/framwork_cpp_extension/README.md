## 概述
本样例展示了自定义算子通过torch原生提供的cppextension方式注册eager模式与torch.compile模式的注册样例，eager模式与torch.compile模式的介绍参考：[Link](https://pytorch.org/get-started/pytorch-2.0)。

## 目录结构介绍
```
├── build_and_run.sh                // 自定义算子wheel包编译安装并执行用例的脚本
├── csrc                            // 算子适配层c++代码目录
│   ├── add_custom.cpp              // 自定义算子正反向适配代码以及绑定
│   ├── function.h                  // 正反向接口头文件
│   ├── pytorch_npu_helper.hpp      // 自定义算子调用和下发框架
│   └── registration.cpp            // 自定义算子aten ir注册文件
├── custom_ops                      // 自定义算子包python侧代码
│   ├── add_custom.py               // 提供自定义算子python调用接口
│   └── __init__.py                 // python初始化文件
├── setup.py                        // wheel包编译文件
└── test                            // 测试用例目录
    ├── test_add_custom_graph.py    // 执行torch.compile模式下用例脚本
    └── test_add_custom.py          // 执行eager模式下算子用例脚本
```

## 样例脚本build_and_run.sh关键步骤解析

  - 编译适配层代码并生成wheel包
    ```bash
    python3 setup.py build bdist_wheel
    ```

  - 安装编译生成的wheel包
    ```bash
    cd ${BASE_DIR}
    pip3 install dist/*.whl
    ```

## 自定义算子入图关键步骤解析
  可以在test_add_custom_graph.py文件查看相关注册实现。
  - 根据Ascend C工程产生的REG_OP算子原型填充torchair.ge.custom_op的参数。

    AddCustom的REG_OP原型为：
    ```cpp
    REG_OP(AddCustom)
        .INPUT(x, ge::TensorType::ALL())
        .INPUT(y, ge::TensorType::ALL())
        .OUTPUT(z, ge::TensorType::ALL())
        .OP_END_FACTORY_REG(AddCustom);
    ```

  - 注册自定义算子converter
    ```python
    from torchair import register_fx_node_ge_converter
    from torchair.ge import Tensor
    @register_fx_node_ge_converter(torch.ops.myops.add_custom.default)
    def convert_npu_add_custom(x: Tensor, y: Tensor, z: Tensor = None, meta_outputs: Any = None):
        return torchair.ge.custom_op(
            "AddCustom",
            inputs={"x": x, "y": y,},
            outputs=['z']
        )
    ```

## 运行样例算子
该样例脚本基于Pytorch2.1、python3.9 运行
### 1.编译算子工程
运行此样例前，请参考[编译算子工程](../README.md#operatorcompile)完成前期准备。

### 2.pytorch调用的方式调用样例运行

  - 进入到样例目录

    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/0_introduction/1_add_frameworklaunch/CppExtensionInvocation
    ```

  - 样例执行

    样例执行过程中会自动生成测试数据，然后运行pytorch样例，最后检验运行结果。具体过程可参见build_and_run.sh脚本。
    ```bash
    bash build_and_run.sh
    ```

#### 其他样例运行说明
  - 环境安装完成后，样例支持单独执行：eager模式与compile模式的测试用例
    - 执行pytorch eager模式的自定义算子测试文件
      ```bash
      python3 test_add_custom.py
      ```
    - 执行pytorch torch.compile模式的自定义算子测试文件
      ```bash
      python3 test_add_custom_graph.py
      ```

### 其他说明
    更加详细的Pytorch适配算子开发指导可以参考[LINK](https://gitee.com/ascend/op-plugin/wikis)中的“算子适配开发指南”。

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/01/17 | 新增本readme |