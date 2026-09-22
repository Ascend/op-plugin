<h1 align="center">OpPlugin</h1>

<p align="center">
  <strong>TorchNPU 的昇腾 NPU 算子适配子仓</strong>
</p>

<p align="center">
  简体中文 | <a href="./README.md">English</a>
</p>

<p align="center">
  <a href="#版本配套">版本配套</a> ·
  <a href="#安装">安装</a> ·
  <a href="#算子开发">算子开发</a> ·
  <a href="#api-参考">API 参考</a> ·
  <a href="#贡献与交流">贡献与交流</a> ·
  <a href="https://gitcode.com/Ascend/pytorch">TorchNPU 主仓</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C++-00599C?style=flat&amp;logo=cplusplus&amp;logoColor=white" alt="C++">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&amp;logo=python&amp;logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Platform-Ascend%20NPU-C31D20" alt="Platform: Ascend NPU">
  <a href="https://gitcode.com/Ascend/pytorch"><img src="https://img.shields.io/badge/TorchNPU-Submodule-blue" alt="TorchNPU submodule"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause-8A2BE2" alt="License: BSD-3-Clause"></a>
  <a href="https://gitcode.com/Ascend/op-plugin"><img src="https://img.shields.io/badge/Repo-GitCode-D71D3A" alt="GitCode repository"></a>
</p>

## 简介

**OpPlugin** 是 [TorchNPU](https://gitcode.com/Ascend/pytorch/blob/master/README.zh.md) 的算子适配子仓，为 PyTorch 原生算子和 TorchNPU 自定义算子提供昇腾 NPU 适配实现。

TorchNPU 通过 `third_party/op-plugin` 子模块集成本仓，将算子适配代码编译到 `torch_npu` 软件包中。OpPlugin 的编译和运行依赖 TorchNPU；使用算子时，安装配套的 TorchNPU 即可，无需单独安装 OpPlugin 软件包。

## 核心功能

- **算子适配：** 对接 CANN 算子接口，实现 PyTorch 原生算子与 TorchNPU 自定义算子的 NPU 计算。
- **配置与代码生成：** 通过 YAML 配置管理算子接口、版本适配及前反向绑定，并支持结构化适配代码生成。
- **开发与验证：** 提供算子适配指南、自定义算子扩展示例和测试用例，支持算子开发与验证。

## 版本配套

OpPlugin 随对应版本的 TorchNPU 集成和发布，环境配套和支持策略统一参考 TorchNPU：

- [版本配套](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)：PyTorch、TorchNPU、CANN 和 Python 的配套关系。
- [支持说明](https://gitcode.com/Ascend/pytorch/blob/master/SUPPORT.md)：版本支持状态与生命周期。

使用历史版本时，请切换到对应分支阅读文档。

## 安装

### 使用已有算子

按照 [TorchNPU 安装指南](https://gitcode.com/Ascend/pytorch/blob/master/README.zh.md#安装)准备驱动、固件、CANN、PyTorch 和 TorchNPU。安装后，可通过 [TorchNPU 快速开始](https://gitcode.com/Ascend/pytorch/blob/master/README.zh.md#快速开始)验证 NPU 环境。

### 源码编译

需要修改或新增算子时，可选择以下方式：

- **从 TorchNPU 主仓构建：** 按照 [TorchNPU 源码安装指南](https://gitcode.com/Ascend/pytorch/blob/master/docs/zh/installation_guide/building_from_source.md)获取源码和子模块，在 `third_party/op-plugin` 中开发并随主仓编译。
- **从 OpPlugin 本仓构建：** 按照 [OpPlugin 源码编译指南](docs/zh/install.md)准备环境，并使用 `ci/build.sh` 与指定版本的 TorchNPU 协同编译。

本仓构建脚本会获取 TorchNPU 源码，将本地算子适配代码集成到其中，最终在 `dist/` 下生成 `torch_npu` wheel 包。Python 版本和 TorchNPU 分支需按目标版本配套选择。

## 算子开发

| 开发任务 | 参考文档 |
| --- | --- |
| 配置算子接口、版本适配、前反向绑定与结构化适配 | [API 适配开发流程](op_plugin/config/README.md) |
| 通过 OpPlugin 适配 Ascend C 自定义算子 | [TorchNPU 算子适配指南](https://gitcode.com/Ascend/pytorch/blob/master/docs/zh/developer_notes/custom_operator_adaptation/opplugin_operator_adaptation/_menu_opplugin_operator_adaptation.md) |
| 通过 C++ extensions 构建自定义算子扩展 | [自定义算子扩展示例](examples/README.md) |
| 查阅和补充算子测试用例 | [测试目录](test) |

## API 参考

[TorchNPU 自定义 API](docs/zh/custom_APIs/menu_Pytorch_API.md)提供接口功能、函数原型、参数说明、支持约束和调用示例。具体接口的版本和硬件支持范围以对应 API 文档为准。

## 目录结构

```text
├── ci                    # 构建与测试脚本
├── codegen               # 算子适配代码生成
├── docs                  # 安装、安全和 API 文档
├── examples              # 自定义算子扩展示例
├── op_plugin             # 算子适配实现
│   ├── config            # 算子接口与适配配置
│   ├── ops               # 算子实现
│   │   ├── aclops        # aclop 算子适配
│   │   └── opapi         # aclnn 算子适配
│   └── python            # Python 相关实现
├── test                  # 算子测试用例
└── torchnpugen            # TorchNPU 代码生成工具
```

## 贡献与交流

OpPlugin 复用 [TorchNPU 贡献指南](https://gitcode.com/Ascend/pytorch/blob/master/CONTRIBUTING.md)中的开发流程和贡献规范。涉及本仓的代码、测试和文档修改，请向 OpPlugin 提交 PR。

算子适配问题或建议请提交 [OpPlugin Issues](https://gitcode.com/Ascend/op-plugin/issues)；TorchNPU 框架相关问题请提交 [TorchNPU Issues](https://gitcode.com/Ascend/pytorch/issues)。

OpPlugin 由 Ascend for PyTorch 社区的 [Core SIG](https://gitcode.com/Ascend/community/tree/master/AscendForPyTorch/sigs/core)负责设计、实现与维护，欢迎参与交流和贡献。

## 安全声明

使用本仓前，请阅读 [OpPlugin 安全声明](docs/zh/SECURITYNOTE.md)。TorchNPU 的通用安全加固与运行要求请参见 [TorchNPU 安全声明](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.md)。

## 免责声明

致OpPlugin插件使用者

- 本插件仅供调试和开发使用，使用者需自行承担使用风险，并理解以下内容：
    - 数据处理及删除：用户在使用本插件过程中产生的数据属于用户责任范畴。建议用户在使用完毕后及时删除相关数据，以防信息泄露。
    - 数据保密与传播：使用者了解并同意不得将通过本插件产生的数据随意外发或传播。对于由此产生的信息泄露、数据泄露或其他不良后果，本插件及其开发者概不负责。
    - 用户输入安全性：用户需自行保证输入的命令行的安全性，并承担因输入不当而导致的任何安全风险或损失。对于输入命令行不当所导致的问题，本插件及其开发者概不负责。
- 免责声明范围：本免责声明适用于所有使用本插件的个人或实体。使用本插件即表示您同意并接受本声明的内容，并愿意承担因使用该功能而产生的风险和责任，如有异议请停止使用本插件。
- 在使用本工具之前，请谨慎阅读并理解以上免责声明的内容。对于使用本插件所产生的任何问题或疑问，请及时联系开发者。

## License

OpPlugin 的使用许可证，请参见 [LICENSE](./LICENSE)。

## 致谢

感谢来自社区的每一个 PR，欢迎开发者向 OpPlugin 贡献代码、测试和文档！
