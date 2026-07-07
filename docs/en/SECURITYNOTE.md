# OpPlugin Security Statement

## System Security Hardening

You are advised to enable **address space layout randomization** (ASLR) (level 2) in the system. Run the following command to enable it:

    echo 2 > /proc/sys/kernel/randomize_va_space

## Recommended Running Users

The execution of OpPlugin depends on `torch_npu`. For details, see [Recommended Running Users](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/SECURITYNOTE.md#running-user-recommendations) in the `torch_npu` repository.

## File Permission Control

1. Control file and directory permissions during installation and use. Secure the permissions by referring to [Recommended Maximum Scenario Permissions for Files and Folders](#11111). To save installation or uninstallation logs, add `--log \<FILE>` to the end of your command, and restrict permissions on the `\<FILE>` file and directory.

2. Set the running system `umask` value to `0027` or higher on both the host and the container. This setting ensures that the default maximum permission is `750` for new folders and `640` for new files.

<h3 id="11111">Recommended Maximum Scenario Permissions for Files and Folders</h3>

|   Type                            |   Maximum Linux Permission  |
|----------------------------------- |-----------------------|
|  Home directory                        |   750 (rwxr-x---)    |
|  Program files (including scripts and library files)      |   550 (r-xr-x---)    |
|  Program file directory                      |   550 (r-xr-x---)    |
|  Configuration files                          |   640 (rw-r-----)    |
|  Configuration file directory                      |   750 (rwxr-x---)    |
|  Log files (recorded or archived)      |   440 (r--r-----)    |
|  Log files (being recorded)                 |   640 (rw-r-----)   |
|  Log file directory                      |   750 (rwxr-x---)    |
|  Debug files                        |   640 (rw-r-----)     |
|  Debug file directory                     |   750 (rwxr-x---)    |
|  Temporary file directory                      |   750 (rwxr-x---)    |
|  Maintenance and upgrade file directory                  |   770 (rwxrwx---)     |
|  Service data files                      |   640 (rw-r-----)     |
|  Service data file directory                  |   750 (rwxr-x---)     |
|  Key components, private keys, certificates, and ciphertext file directory  |   700 (rwx------)     |
|  Key components, private keys, certificates, and ciphertext files      |   600 (rw-------)    |
|  APIs and scripts for encryption and decryption             |   500 (r-x------)     |

## Debugging Tool Statement

The execution of OpPlugin depends on `torch_npu`. For details, see [Torch NPU Repository Debugging Tool Statement](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/SECURITYNOTE.md#debugging-tool-statement) in the `torch_npu` repository.

## Data Security Statement

The execution of OpPlugin depends on `torch_npu`. For details, see [Data Security Statement](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/SECURITYNOTE.md#data-security-statement) in the `torch_npu` repository.

## Build Security Statement

The execution of OpPlugin depends on `torch_npu`. For details, see [Build Security Statement](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/SECURITYNOTE.md#build-security-statement) in the `torch_npu` repository.

## Runtime Security Statement

The execution of OpPlugin depends on `torch_npu`. For details, see [Runtime Security Statement](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/SECURITYNOTE.md#runtime-security-statement) in the `torch_npu` repository.

## Public IP Address Statement

Public IP addresses are used in the configuration files and scripts of OpPlugin. For details, see [Public IP Addresses](#public-ip-addresses).

### Public IP Addresses

|   Type  |   Open-Source Code Address  |   File  | Public IP Address/Public URL/Domain Name/Email Address                                                                                    |   Description  |
|------------------------|-------------------------|-------------------------|------------------------------------------------------------------------------------------------------------|-------------------------|
|   Development introduction |   N/A  |   ci\build.sh   | [https://gitcode.com/ascend/pytorch.git](https://gitcode.com/ascend/pytorch.git)                                                                     |   The build script pulls code from the `torch_npu` repository address to execute the build process.  |
|   Development introduction |   N/A  |   ci\exec_ut.sh   | [https://gitcode.com/ascend/pytorch.git](https://gitcode.com/ascend/pytorch.git)                                                                     |   The `UT` script pulls code from the `torch_npu` repository address to perform `UT` tests.  |
| Open-source code introduction|pytorch\aten\src\ATen\native\TensorCompare.cpp  | op_plugin\ops\opapi\IsInKernelNpuOpApi.cpp | [https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/arraysetops.py#L575](https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/arraysetops.py#L575) | The algorithm implementation references this NumPy source code URL.|

## Public API Statement

The execution of OpPlugin depends on `torch_npu`, and no public API is provided.

## Communication Security Hardening

The execution of OpPlugin depends on `torch_npu`. For details, see [Communication Security Hardening](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/SECURITYNOTE.md#communication-security-hardening) in the `torch_npu` repository.

## Communication Matrix

The execution of OpPlugin depends on `torch_npu`. For details, see [Communication Matrix](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/SECURITYNOTE.md#communication-matrix) in the `torch_npu` repository.
