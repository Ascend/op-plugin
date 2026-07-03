# Installing OpPlugin by Building from Source

## Installation Description

1. Hardware Support
 
   The following table lists the Ascend training devices that can serve as training environments for PyTorch-based models.
       
   | Product Series              | Product Model                        |
   |-----------------------|----------------------------------|
   | Atlas training products    | Atlas 800 training server (model 9000)|
   |                       | Atlas 800 training server (model 9010)|
   |                       | Atlas 900 PoD (model 9000)      |
   |                       | Atlas 300T training card (model 9000)   |
   |                       | Atlas 300T Pro training card (model 9000)|
   | Atlas A2 training products | Atlas 800T A2 training server         |
   |                       | Atlas 900 A2 PoD cluster basic unit    |
   |                       | Atlas 200T A2 Box16 heterogeneous subrack     |
   | Atlas A3 training products | Atlas 800T A3 training server         |
   |                       | Atlas 900 A3 SuperPoD server    |
    
   The following table lists the Ascend inference devices that can serve as inference environments for foundation models.
       
   | Product Series              | Product Model                        |
   |-----------------------|----------------------------------|
   | Atlas 800I A2 inference products | Atlas 800I A2 inference server         |

2. Software Support

    <a id="table1"></a>

   | PyTorch | Ascend Extension for PyTorch | OpPlugin | Python                      | GCC  |
   |---------|------------------------------|----------|-----------------------------|------|
   | 2.7.1   | v2.7.1                       | master   | 3.9, 3.10, 3.11, 3.12, 3.13 | 11.2 |
   | 2.8.0   | v2.8.0                       | master   | 3.9, 3.10, 3.11, 3.12, 3.13 | 13.3 |
   | 2.9.0   | v2.9.0                       | master   | 3.10, 3.11, 3.12, 3.13      | 13.3 |
   | 2.10.0  | v2.10.0                      | master   | 3.10, 3.11, 3.12, 3.13      | 13.3 |
   | 2.11.0  | v2.11.0                      | master   | 3.10, 3.11, 3.12, 3.13      | 13.3 |
   | 2.12.0  | v2.12.0                      | master   | 3.10, 3.11, 3.12, 3.13      | 13.3 |
   | 2.13.0  | master                       | master   | 3.10, 3.11, 3.12, 3.13      | 13.3 |

## Installing Dependencies

 Install the system dependencies and the official PyTorch framework. You are advised to use the Docker image provided by `torch_npu` for building. For dependency installation and image usage guides, see [Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch/tree/v2.7.1-7.3.0#%E4%BD%BF%E7%94%A8%E6%BA%90%E4%BB%A3%E7%A0%81%E8%BF%9B%E8%A1%8C%E5%AE%89%E8%A3%85).
 
## Procedure
 
1. Run the following command to set the CANN environment variables:
 
   ```bash
   source <CANN_install_dir>/<CANN_path>/set_env.sh
   ```
 
   The default path of the environment variable script is typically `/usr/local/npu/ascend-toolkit/set_env.sh`, where `ascend-toolkit` depends on the installed CANN package name.
 
2. Build the OpPlugin binary package.
 
   Download the source code for the target `op-plugin` branch and navigate to its root directory:

   ```bash
   git clone --branch master https://gitcode.com/ascend/op-plugin.git
   cd op-plugin
   ```

   Run the build script to generate the binary installation package. The following example uses `PyTorch 2.10.0`:

   ```bash
   bash ci/build.sh --python=3.9 --pytorch=v2.10.0
   ```

    > [!NOTICE] 
    > Ensure that the GCC and Python versions meet the requirements listed in [Software Support](#table1).
    > During the build process, the system creates a `build` directory in the OpPlugin root directory and downloads the corresponding `torch_npu` source code to build it alongside OpPlugin. If the `build/pytorch` directory exists, the system does not download the `torch_npu` source code again. To download the latest `torch_npu` source code, delete the `build/pytorch` directory.
 
3. After the build is complete, install the generated `torch_npu` package in the `dist` directory. If you install the package as a non-root user, add `--user` to the end of the command.
 
   ```bash
   pip3 install --upgrade dist/torch_npu-{torch_npu_version}-{Python_version}-{arch}.whl
   # Replace placeholders with the actual names from the generated .whl package. {torch_npu_version} indicates the compiled torch_npu version, {Python_version} indicates the Python version used, and {arch} indicates the target architecture.
   # Example: torch_npu-2.7.1.post13-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
   ```

## Uninstallation

 Run the following command to uninstall `torch`:

 ```bash
 pip uninstall torch_npu
 ```
