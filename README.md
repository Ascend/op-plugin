# OpPlugin

## 简介

本项目开发了Ascend Extension for Pytorch（torch_npu）算子插件，为使用PyTorch框架的开发者提供便捷的NPU算子库调用能力。
OpPlugin算子插件的编译、使用依赖昇腾Ascend Extension for PyTorch。因此，在编译OpPlugin之前，需要了解、安装昇腾PyTorch。使用手册可参考昇腾社区[Ascend Extension for Pytorch](https://gitcode.com/ascend/pytorch/blob/master/README.zh.md)。

## 目录结构

关键目录如下:

```ColdFusion
├─docs                             # 文档目录
├─ci                               # 自动化构建与测试脚本目录
├─op_plugin                        # 项目核心目录
│  ├─config                        # 配置管理目录
│  ├─ops                           # 算子实现目录
│  ├─python                        # python绑定目录
├─codegen                          # 代码生成目录
├─examples                         # 示例目录
└─test                             # 测试目录
```

## 版本配套表

op-plugin仓旨在为**torch_npu**提供运行所需要的算子适配文件，两个仓的对应关系如下：

| op-plugin分支 | 对应Ascend Extension for PyTorch版本 |
| ------------- | :----------------------------------: |
| master       |     主线版本，如v2.7.1等              |
| 26.0.0       |    26.0.0版本，如v2.7.1-26.0.0等      |
| 7.3.0        |     7.3.0版本，如v2.7.1-7.3.0等       |
| 7.2.0        |     7.2.0版本，如v2.7.1-7.2.0等       |
| 7.1.0        |     7.1.0版本，如v2.1.0-7.1.0等       |
| 7.0.0        |     7.0.0版本，如v2.1.0-7.0.0等       |
| 6.0.0        |     6.0.0版本，如v2.1.0-6.0.0等       |
| 6.0.rc3      |   6.0.rc3版本，如v2.1.0-6.0.rc3等     |
| 6.0.rc2      |   6.0.rc2版本，如v2.1.0-6.0.rc2等     |
| 6.0.rc1      |   6.0.rc1版本，如v2.1.0-6.0.rc1等     |
| 5.0.0        |     5.0.0版本，如v2.1.0-5.0.0等       |
| 5.0.rc3      |   5.0.rc3版本，如v2.1.0-5.0.rc3等     |

## 安装OpPlugin

支持通过源码编译的方式安装OpPlugin。具体操作，请参考[安装OpPlugin](docs/zh/install.md)。

## 快速入门

提供了一个通过OpPlugin插件实现PyTorch调用Ascend C自定义算子的完整开发指南，涵盖了从环境配置、算子注册、适配实现到测试验证提供了全流程说明。具体操作，请参考[调用样例](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/framework_feature_guide_pytorch/opplugin_operator_adaptation.md)。

## API参考

基于PyTorch2.10.0/2.9.0/2.8.0/2.7.1版本，提供Ascend Extension for PyTorch自定义API的功能说明、函数原型、参数说明与调用示例等。具体信息，请参考[自定义API参考](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/menu_Pytorch_API.md)。

## 生命周期

op-plugin仓依赖**torch_npu**运行，生命周期请参考**torch_npu**中的[PyTorch版本维护策略](https://gitcode.com/ascend/pytorch/blob/master/README.zh.md#pytorch%E7%89%88%E6%9C%AC%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5)。

## 贡献指导

介绍如何向OpPlugin仓库贡献代码，具体请参见[贡献指南](https://gitcode.com/Ascend/pytorch/blob/master/CONTRIBUTING.md)。

## 联系我们

如果有任何疑问或建议，请提交[GitCode Issues](https://gitcode.com/Ascend/pytorch/issues)，我们会尽快回复。感谢您的支持。

## 安全声明

主要描述了OpPlugin的安全加固信息、公网地址信息及通信矩阵等内容。具体介绍，请参考[OpPlugin安全声明](docs/zh/SECURITYNOTE.md)。

## 免责声明

致OpPlugin插件使用者

- 本插件仅供调试和开发使用，使用者需自行承担使用风险，并理解以下内容：
    - 数据处理及删除：用户在使用本插件过程中产生的数据属于用户责任范畴。建议用户在使用完毕后及时删除相关数据，以防信息泄露。
    - 数据保密与传播：使用者了解并同意不得将通过本插件产生的数据随意外发或传播。对于由此产生的信息泄露、数据泄露或其他不良后果，本插件及其开发者概不负责。
    - 用户输入安全性：用户需自行保证输入的命令行的安全性，并承担因输入不当而导致的任何安全风险或损失。对于输入命令行不当所导致的问题，本插件及其开发者概不负责。
- 免责声明范围：本免责声明适用于所有使用本插件的个人或实体。使用本插件即表示您同意并接受本声明的内容，并愿意承担因使用该功能而产生的风险和责任，如有异议请停止使用本插件。
- 在使用本工具之前，请谨慎阅读并理解以上免责声明的内容。对于使用本插件所产生的任何问题或疑问，请及时联系开发者。

  ## License

  OpPlugin的使用许可证，详见[LICENSE](http://gitcode.com/Ascend/op-plugin/blob/master/LICENSE)。

  ## 致谢

  感谢来自社区的每一个PR，欢迎贡献Ascend Extension for TensorPipe插件！
