# OpPlugin

<p>
    English | <a href="./README.zh.md">简体中文</a>
</p>

## Brief Introduction 

In this project, the TorchNPU operator plug-in is developed to provide the NPU operator library invoking capability for developers using the PyTorch framework. The compilation and use of the OpPlugin operator plug-in depend on the Ascend TorchNPU. Therefore, you need to understand and install Ascend PyTorch before compiling OpPlugin. For details about the user manual, see the Ascend community.[TorchNPU](https://gitcode.com/ascend/pytorch/blob/master/README.md).

## Directory structure

The key directories are as follows:

```text
├─docs                             #Document Directory
├─ci                               #Directory for storing the automatic build and test scripts.
├─op_plugin                        #Project core catalog
│  ├─config                        #Configuration Management Directory
│  ├─ops                           #Operator Implementation Directory
│  ├─python                        #Directory bound with Python
├─codegen                          #Code generation directory
├─examples                         #Sample Directory
└─test                             #Test Directory
```

## Version mapping table

The OpPlugin repository provides the operator adaptation files required by the TorchNPU. The mapping between the two repositories is as follows:

| OpPlugin Branch |      Corresponding TorchNPU version      |
|:--------------- |:----------------------------------------:|
| master          |    Mainline version, such as v2.7.1.     |
| 26.1.0          |  Version 26.1.0, such as v2.7.1-26.1.0   |
| 26.0.0          |  Version 26.0.0, such as v2.7.1-26.0.0   |
| 7.3.0           |       7.3.0, such as v2.7.1-7.3.0        |
| 7.2.0           |   7.2.0 version, such as v2.7.1-7.2.0    |
| 7.1.0           |   7.1.0 version, such as v2.1.0-7.1.0    |
| 7.0.0           |     7.0.0, for example, v2.1.0-7.0.0     |
| 6.0.0           |     6.0.0, for example, v2.1.0-6.0.0     |
| 6.0.rc3         | Version 6.0.rc3, such as v2.1.0-6.0.rc3. |
| 6.0.rc2         | Version 6.0.rc2, such as v2.1.0-6.0.rc2. |
| 6.0.rc1         | Version 6.0.rc1, such as v2.1.0-6.0.rc1  |
| 5.0.0           |   Version 5.0.0, such as v2.1.0-5.0.0    |
| 5.0.rc3         | 5.0.rc3 version, such as v2.1.0-5.0.rc3  |

## Installing OpPlugin

OpPlugin can be installed by compiling the source code. For details, see.[Installing OpPlugin](docs/en/install.md).

## Quick Start

This document provides a complete development guide for PyTorch to invoke Ascend C custom operator by using the OpPlugin plug-in, covering the entire process from environment configuration, operator registration, adaptation implementation, to test and verification. For details, see the Invoking Example.

## API Reference

Provides the function description, function prototype, parameter description, and invoking examples of the TorchNPU customized API based on the PyTorch2.10.0/2.9.0/2.8.0/2.7.1 version. For details, see the Custom API.

## Life Cycle

The OpPlugin repository depends on the TorchNPU. For details about the life cycle, see the [PyTorch Version Maintenance Policy](https://gitcode.com/ascend/pytorch/blob/master/README.md).

## Contribution guidance

This section describes how to contribute code to the OpPlugin repository. For details, see the [Contribution Guide](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/zh/CONTRIBUTING.md).

## Contact us

If you have any questions or suggestions, please submit [GitCode Issues](https://gitcode.com/Ascend/pytorch/issues) We'll get back to you as soon as we can. Thank you for your support.

## Safety Statement

This document describes the security hardening information, public network address information, and communication matrix of OpPlugin. For details, see the OpPlugin Security Statement.
<!-- 
 [OpPlugin Security Statement](docs/en/SECURITYNOTE.en.md). -->

## Disclaimer

To OpPlugin plug-in users

- This plug-in is for debugging and development only. You must bear the risks and understand the following:
    
    - Data processing and deletion: The data generated during the use of this plug-in belongs to the user's responsibility. You are advised to delete related data in time after using the data to prevent information leakage.
    - Data confidentiality and dissemination: Users understand and agree not to send or disseminate the data generated through this plug-in at will. This plug-in and its developers are not responsible for any information leakage, data leakage, or other adverse consequences arising therefrom.
    - User input security: Users must ensure the security of the entered command lines and bear any security risks or losses caused by improper input. This plug-in and its developers are not responsible for any problems caused by improper command line input.
- Scope of Disclaimer: This disclaimer applies to all individuals or entities using this plug-in. By using this plug-in, you agree to and accept the content of this statement and are willing to bear the risks and responsibilities arising from the use of this function. If you have any objection, please stop using this plug-in.
- Read and understand the disclaimer before using this tool. For any questions or questions arising from the use of this plug-in, please contact the developer in time.
    
## License
    
OpPlugin license. For details, see the[LICENSE](http://gitcode.com/Ascend/op-plugin/blob/master/LICENSE).
    
## Acknowledgment
    
Thank you for every PR from the community. Welcome to contribute the Ascend Extension for TensorPipe plug-in!
