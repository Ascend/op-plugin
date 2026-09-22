<h1 align="center">OpPlugin</h1>

<p align="center">
  <strong>Ascend NPU Operator Adaptation Sub-repository for TorchNPU</strong>
</p>

<p align="center">
  English | <a href="./README.zh.md">简体中文</a>
</p>

<p align="center">
  <a href="#compatibility">Compatibility</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#operator-development">Operator Development</a> ·
  <a href="#api-reference">API Reference</a> ·
  <a href="#contributing-and-community">Contributing</a> ·
  <a href="https://gitcode.com/Ascend/pytorch">TorchNPU Repository</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C++-00599C?style=flat&amp;logo=cplusplus&amp;logoColor=white" alt="C++">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&amp;logo=python&amp;logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Platform-Ascend%20NPU-C31D20" alt="Platform: Ascend NPU">
  <a href="https://gitcode.com/Ascend/pytorch"><img src="https://img.shields.io/badge/TorchNPU-Submodule-blue" alt="TorchNPU submodule"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause-8A2BE2" alt="License: BSD-3-Clause"></a>
  <a href="https://gitcode.com/Ascend/op-plugin"><img src="https://img.shields.io/badge/Repo-GitCode-D71D3A" alt="GitCode repository"></a>
</p>

## Overview

**OpPlugin** is the operator adaptation sub-repository of [TorchNPU](https://gitcode.com/Ascend/pytorch/blob/master/README.md). It provides Ascend NPU implementations for native PyTorch operators and TorchNPU custom operators.

TorchNPU integrates this repository through the `third_party/op-plugin` submodule and compiles its operator adaptation code into the `torch_npu` package. Building and running OpPlugin depend on TorchNPU. To use these operators, install a compatible TorchNPU package; no separate OpPlugin package is required.

## Key Features

- **Operator adaptation:** Connects to CANN operator interfaces to implement NPU computation for native PyTorch operators and TorchNPU custom operators.
- **Configuration and code generation:** Uses YAML configuration to manage operator interfaces, version adaptation, and forward/backward bindings, with support for structured adaptation code generation.
- **Development and verification:** Provides operator adaptation guides, custom operator extension examples, and test cases.

## Compatibility

OpPlugin is integrated and released with the corresponding TorchNPU version. Refer to TorchNPU for shared environment requirements and support policies:

- [Compatibility](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md): Version mappings for PyTorch, TorchNPU, CANN, and Python.
- [Support](https://gitcode.com/Ascend/pytorch/blob/master/SUPPORT.en.md): Version support status and lifecycle.

For historical versions, read the documentation on the corresponding branch.

## Installation

### Using Existing Operators

Follow the [TorchNPU installation guide](https://gitcode.com/Ascend/pytorch/blob/master/README.md#installation) to prepare the driver, firmware, CANN, PyTorch, and TorchNPU. Then verify your NPU environment with the [TorchNPU quick start](https://gitcode.com/Ascend/pytorch/blob/master/README.md#quick-start).

### Building from Source

To modify or add operators, choose one of the following workflows:

- **Build from the TorchNPU repository:** Follow the [TorchNPU source installation guide (Chinese)](https://gitcode.com/Ascend/pytorch/blob/master/docs/zh/installation_guide/building_from_source.md) to obtain the source and submodules, develop in `third_party/op-plugin`, and build with TorchNPU.
- **Build from this repository:** Follow the [OpPlugin source build guide](docs/en/install.md) to prepare the environment, then use `ci/build.sh` to build with a specified TorchNPU version.

The build script in this repository obtains the TorchNPU source, integrates the local operator adaptation code, and produces a `torch_npu` wheel in `dist/`. Select the Python version and TorchNPU branch according to the target version's requirements.

## Operator Development

| Task | Documentation |
| --- | --- |
| Configure operator interfaces, version adaptation, forward/backward bindings, and structured adaptation | [API adaptation workflow (Chinese)](op_plugin/config/README.md) |
| Adapt Ascend C custom operators through OpPlugin | [TorchNPU operator adaptation guide](https://gitcode.com/Ascend/pytorch/blob/master/docs/en/developer_notes/framework_feature_guide_pytorch/opplugin_operator_adaptation.md) |
| Build custom operator extensions using C++ extensions | [Custom operator extension examples (Chinese)](examples/README.md) |
| Review and add operator test cases | [Tests](test) |

## API Reference

The [TorchNPU custom API reference (Chinese)](docs/zh/custom_APIs/menu_Pytorch_API.md) describes API functionality, signatures, parameters, constraints, and usage examples. Refer to each API document for its supported versions and hardware.

## Directory Structure

```text
├── ci                    # Build and test scripts
├── codegen               # Operator adaptation code generation
├── docs                  # Installation, security, and API documentation
├── examples              # Custom operator extension examples
├── op_plugin             # Operator adaptation implementations
│   ├── config            # Operator interface and adaptation configuration
│   ├── ops               # Operator implementations
│   │   ├── aclops        # aclop operator adaptation
│   │   └── opapi         # aclnn operator adaptation
│   └── python            # Python-related implementations
├── test                  # Operator test cases
└── torchnpugen            # TorchNPU code generation tools
```

## Contributing and Community

OpPlugin follows the development workflow and contribution conventions in the [TorchNPU contribution guide](https://gitcode.com/Ascend/pytorch/blob/master/CONTRIBUTING.en.md). Submit changes to this repository's code, tests, and documentation as OpPlugin PRs.

Report operator adaptation issues or suggestions through [OpPlugin Issues](https://gitcode.com/Ascend/op-plugin/issues). For TorchNPU framework issues, use [TorchNPU Issues](https://gitcode.com/Ascend/pytorch/issues).

The Ascend for PyTorch community's [Core SIG](https://gitcode.com/Ascend/community/tree/master/AscendForPyTorch/sigs/core) is responsible for the design, implementation, and maintenance of OpPlugin. Contributions and discussions are welcome.

## Security Note

Read the [OpPlugin security statement](docs/en/SECURITYNOTE.md) before using this repository. For shared security hardening and runtime requirements, see the [TorchNPU security note](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.en.md).

## Disclaimer

To OpPlugin plug-in users

- This plug-in is for debugging and development only. You must bear the risks and understand the following:
    
    - Data processing and deletion: The data generated during the use of this plug-in belongs to the user's responsibility. You are advised to delete related data in time after using the data to prevent information leakage.
    - Data confidentiality and dissemination: Users understand and agree not to send or disseminate the data generated through this plug-in at will. This plug-in and its developers are not responsible for any information leakage, data leakage, or other adverse consequences arising therefrom.
    - User input security: Users must ensure the security of the entered command lines and bear any security risks or losses caused by improper input. This plug-in and its developers are not responsible for any problems caused by improper command line input.
- Scope of Disclaimer: This disclaimer applies to all individuals or entities using this plug-in. By using this plug-in, you agree to and accept the content of this statement and are willing to bear the risks and responsibilities arising from the use of this function. If you have any objection, please stop using this plug-in.
- Read and understand the disclaimer before using this tool. For any questions or questions arising from the use of this plug-in, please contact the developer in time.

## License

See [LICENSE](./LICENSE) for the OpPlugin license.

## Acknowledgments

Thanks for every PR from the community! Contributions to OpPlugin code, tests, and documentation are welcome.
