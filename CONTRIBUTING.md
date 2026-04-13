# OpPlugin 贡献指南

感谢您考虑为 OpPlugin 做出贡献！我们欢迎任何形式的贡献，包括错误修复、功能增强、文档改进等。无论您是经验丰富的开发者还是第一次参与开源项目，您的帮助都是非常宝贵的。

## 项目介绍

OpPlugin 是 Ascend Extension for PyTorch（torch_npu）的算子插件，为使用 PyTorch 框架的开发者提供便捷的 NPU 算子库调用能力。本项目旨在为 torch_npu 提供运行所需要的算子适配文件。

### 项目架构

```text
op-plugin
├── ci/                            # CI 构建脚本
│── docs/                          # 项目文档
├── op_plugin/                     # 核心算子插件实现
│   ├── ops/                       # 算子实现
│   │   ├── aclops/                # ACL 算子内核实现
│   │   ├── atb/                   # ATB 算子实现
│   │   └── opapi/                 # OpAPI 算子封装
│   ├── python/                    # Python 接口
│   ├── config/                    # 算子配置
│   ├── utils/                     # 工具类
│   └── third_party/               # 第三方依赖
├── examples/                      # 示例代码
│   ├── aclnn_extension/           # ACLNN 扩展示例
│   ├── cpp_extension/             # C++ 扩展示例
│   └── framwork_cpp_extension/    # 框架 C++ 扩展示例
├── torchnpugen/                   # 代码生成工具
│   ├── struct/                    # 数据结构生成
│   ├── api/                       # API 生成
│   └── templates/                 # 代码模板
└── test/                          # 测试用例
```

### 核心模块说明

| 模块 | 说明 |
|------|------|
| `op_plugin/ops/aclops` | ACL 算子内核实现：基于 AscendCL 的算子底层实现 |
| `op_plugin/ops/atb` | ATB 算子实现：Transformer 加速库算子封装 |
| `op_plugin/ops/opapi` | OpAPI 算子封装：PyTorch 算子接口适配层 |
| `op_plugin/config` | 算子配置：derivatives.yaml 定义算子求导规则 |
| `torchnpugen` | 代码生成工具：自动生成算子适配代码 |
| `test/test_base_ops` | 算子测试：覆盖各算子的功能验证 |

## 贡献方式

我们热情期待您的加入！每一个贡献都是推动 OpPlugin 进步的重要力量：

- **反馈问题**：报告 Bug 或提交功能建议，帮助我们发现并解决问题
- **贡献代码**：提交代码修复或新功能实现，直接参与项目开发
- **完善文档**：改进文档或补充缺失内容，提升项目可读性
- **代码审查**：审查 Pull Request，帮助提升代码质量
- **分享传播**：在博客、社交媒体上分享项目，给仓库点个 ⭐

## 贡献场景

本项目热烈欢迎各种形式的贡献，期待您的参与！

### 一、需求与功能建议

如果您有新功能建议或性能优化想法，我们热情邀请您提交 Issue 与社区深入讨论。

**Issue 类型**：需求/功能建议

**需要包含的内容**：

- **功能背景**：该功能解决什么问题、能为用户带来什么价值
- **功能描述**：详细描述建议的功能
- **设计方案**：技术思路、关键模块设计、上下游组件关系
- **预期收益**：功能目标、性能指标、精度表现

### 二、Bug 反馈与修复

如果您发现 Bug 或文档问题，我们真诚欢迎您的反馈和修复建议。

**Bug Report 格式**：

- **环境信息**：OpPlugin 版本、torch_npu 版本、Python 版本、CANN 版本等
- **问题描述**：添加标签以便在问题仪表板上突出显示
- **复现步骤**：尽可能详细地描述如何重现问题
- **预期行为**：描述您预期发生的行为
- **给审稿人的特别说明**：如有任何特殊情况

**修复流程**：

1. 在 Issue 中找到对应的 Bug 描述
2. 评论 `/assign` 认领该任务
3. 创建分支进行修复
4. 提交 Pull Request

### 三、协助社区建设

如果您有能力解决他人提出的问题，我们热烈期待您在 Issue 中分享您的解决方案。

## 贡献流程

### 贡献者许可协议

在您第一次向 OpPlugin 社区提交代码之前，需要签署 CLA。

对于个人贡献者，详细信息请参考 [ICLA 在线文档](https://www.mindspore.cn/icla)。

### 开发与测试

1. **Fork 仓库**：在 GitCode 平台点击仓库右上角 "Fork" 按钮，将仓库克隆到个人账户

2. **克隆到本地**：

    ```bash
    git clone https://gitcode.com/<your-username>/op-plugin.git
    cd op-plugin
    ```

3. **创建开发分支**：

    ```bash
    git checkout -b {new_branch_name} origin/master
    ```

4. **代码开发**：请遵循 **[代码规范](#代码规范)**

5. **代码测试**：运行测试确保代码功能正常

6. **门禁检查**：运行 CI 检查，确保代码通过编译、静态检查、UT 测试

7. **提交 Pull Request**：提交 PR 并等待代码审查

8. **社区评审**：如果涉及 patch、头文件宏、API 接口等更新，需提交社区评审

### 代码合入评审要求

以下类型的修改需要社区评审：

- **Patch 替换**：对 PyTorch 原生接口的 patch 替换
- **头文件宏更新**：新增或修改宏定义
- **API 接口变更**：新增、修改或删除公共 API
- **核心组件变更**：内存管理、设备管理等核心模块的修改

**计算类新增 API 接口约束**：

对于计算类新增 API 接口，提交者需要额外提供精度测试结果，确保新接口的计算精度符合要求。精度测试应包括：

- 与 PyTorch CPU/GPU后端的对比结果
- 误差范围统计（最大误差、平均误差、均方根误差等）
- 典型场景验证（如神经网络模型中的实际表现）

### 验收标准

#### API 功能标准

- **入参全覆盖**：所有入参类型均需覆盖，包括必填参数、可选参数、默认值场景
- **测试方法**：通过等价类划分、边界值分析等测试方法，覆盖正常传参与异常传参场景
- **异常处理**：验证异常场景的报错信息是否准确、友好

#### API 精度标准

- **测试用例数量**：每个 dtype 泛化 100~200 个测试 case
- **算子覆盖**：正向算子和反向算子分别按照 2 个 API 生成测试用例
- **精度阈值**：计算结果误差需在允许范围内，确保与 CPU/GPU 结果一致

### 开发交付件

算子适配开发需交付以下内容：

| 交付件 | 说明 |
|--------|------|
| **yaml 配置** | 在 `op_plugin_functions.yaml` 中声明算子版本、schema、适配方式 |
| **前反向绑定配置** | 在 `derivatives.yaml` 中配置算子的前反向绑定关系（仅需前反向绑定的算子） |
| **算子适配代码** | 在 `op_plugin/ops/opapi/` 或 `op_plugin/ops/aclops/` 目录下实现算子接口 |
| **单元测试用例** | 在 `test/test_base_ops/` 目录下编写功能测试和精度测试 |
| **接口文档** | 更新算子对外接口文档（自动生成或手动补充） |

## 代码规范

请遵循这些风格，使 OpPlugin 易于开发、审查和维护。

### 编码指南

- **Python**：建议使用 [PEP 8 编码样式](https://pep8.org/)
- **C++**：建议使用 [Google C++ 编码指南](http://google.github.io/styleguide/cppguide.html)

可使用工具检查代码格式：[CppLint](https://github.com/cpplint/cpplint)、[CppCheck](http://cppcheck.sourceforge.net/)、[pylint](https://pylint.org/)

### 单元测试指南

- **Python**：建议使用 [pytest](http://pytest.org/en/latest/)
- **C++**：建议使用 [Googletest Primer](https://github.com/google/googletest/blob/master/docs/primer.md)

测试用例的设计意图应该通过它的注释名称来反映。

### 重构指南

我们鼓励开发人员重构代码以消除代码异味。所有的代码都应该符合编码风格和测试风格的需求。

## 实操指南

### 环境搭建与编译

**编译构建**：

```bash
# 安装依赖并编译
bash ci/build.sh

# 或使用 CMake 手动编译
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### PR 合入要求

**合入检查清单**（详细要求参考 [PR 模板](./.gitcode/PULL_REQUEST_TEMPLATE.md)）：

- [ ] 代码编译通过
- [ ] 静态检查通过（CppLint、CppCheck 等）
- [ ] UT 测试用例通过
- [ ] 代码风格符合规范（PEP 8、Google C++ Style）
- [ ] 提交信息规范（符合 Conventional Commits）
- [ ] PR 标题正确使用类型标签（feat、fix、refactor、docs、test 等）
- [ ] 代码注释完备，正确记录错误日志
- [ ] 代码实现进行了返回值、空指针等校验
- [ ] 计算类 API 新增需提供精度测试结果

### 功能验证指导

**测试用例位置**：

- `test/test_base_ops/` - 算子基础功能测试
- `test/core_tests/` - 核心功能测试

**运行测试**：

```bash
# 安装测试依赖
pip3 install -r test/requirements.txt

# 运行单个测试文件
python test/test_base_ops/test_add.py

# 或使用 run_test.py
python test/run_test.py -i test_add
```

### 门禁异常处理

门禁异常主要包含如下几种，请根据相关提示解决：

- **编译异常**：请检查代码编译失败的原因，解决问题后重新编译
- **静态检查异常**：请依照提示查找代码中的问题并解决（如代码风格、潜在 Bug 等）
- **UT 测试未通过**：请根据提示查找测试用例不通过项并检查原因

## 提交 Pull Request

1. **推送代码到远程仓库**：

    ```bash
    git add .
    git status
    git commit -m "Your commit title"
    git commit -s --amend  # 添加详细描述
    git push origin {new_branch_name}
    ```

2. **创建 Pull Request**

在 GitCode 上创建 Pull Request，根据 [PR 模板](./.gitcode/PULL_REQUEST_TEMPLATE.md) 完整填写：

- 合入来源
- 修改方案
- 资料变更
- 接口变更
- 功能验证
- CheckList

确认信息完整准确后提交 Pull Request，等待代码审查。

## 社区准则

### 行为准则

我们致力于为所有参与者提供一个友好、安全、包容的环境：

- **尊重差异**：尊重不同的观点和经验，包容多元文化
- **开放心态**：接受建设性的批评，持续学习和进步
- **聚焦贡献**：关注对社区最有利的事情，推动项目发展
- **同理心**：对其他社区成员表示同理心，互帮互助

### 沟通渠道

我们为您提供多种沟通渠道，方便您参与社区互动：

- **[Issues](https://gitcode.com/Ascend/op-plugin/issues)**：用于报告 Bug、提出功能建议
- **[Pull Requests](https://gitcode.com/Ascend/op-plugin/pulls)**：用于代码审查和讨论

### 问题咨询

我们热烈欢迎每一位开发者积极参与社区讨论！期待与您共同成长：

- **发现未解决的问题**：欢迎在 Issue 中发表评论，展示您的解决方案
- **遇到长期未处理的问题**：建议在解决前进行预检查，避免重复工作
- **成功解决了自己报告的问题**：也请分享您的解决方案，让社区一起学习和进步

有任何疑问，随时欢迎在社区中交流讨论，期待您的精彩贡献！
