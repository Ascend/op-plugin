## 概述
本样例展示了自定义算子通过torch原生提供的cppextension方式注册eager模式与torch.compile模式的注册样例，eager模式与torch.compile模式的介绍参考：[Link](https://pytorch.org/get-started/pytorch-2.0)。

## 目录结构介绍
```
├── build_and_run.sh                // 自定义算子wheel包编译安装并执行用例的脚本
├── csrc                            // 算子适配层c++代码目录
│   └── add_custom.cpp              // 自定义算子正反向适配代码、aten ir注册以及绑定
├── cpp_extension_base              // 自定义算子包python侧代码
│   ├── ops.py                      // 定义ops调用
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
  - 环境安装完成后，样例执行：eager模式与compile模式的测试用例
      - 执行pytorch eager模式的自定义算子测试文件
        ```bash
        python3 test_add_custom.py
        ```
      - 执行pytorch torch.compile模式的自定义算子测试文件
        ```bash
        python3 test_add_custom_graph.py
        ```

## 运行样例算子
该样例脚本基于Pytorch2.7、python3.9 运行
  - 执行build_and_run.sh运行编译、安装及测试
    ```bash
    bash build_and_run.sh
    ```

### 其他说明
  - 涉及多卡场景时，需在适配代码中加device_guard保障在正常的device上访问。
  - 更加详细的Pytorch适配算子开发指导可以参考[LINK](https://gitee.com/ascend/op-plugin/wikis)中的“算子适配开发指南”。

## 更新说明
| 时间         | 更新事项     |
|------------| ------------ |
| 2025/03/17 | 新增本readme |