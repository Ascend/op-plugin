# Building and Installing OpPlugin from Source

## Installation Instructions

1. Hardware compatibility
 
   Ascend training devices include the following models, all of which can be used as the training environment for PyTorch models.
       
   | Product Series | Product Model |
   |-----------------------|----------------------------------|
   | Atlas training products | Atlas 800 training server (model 9000) |
   |  | Atlas 800 training server (model 9010) |
   |  | Atlas 900 PoD (model 9000) |
   |  | Atlas 300T training card (model 9000) |
   |  | Atlas 300T Pro training card (model 9000) |
   | Atlas A2 training products | Atlas 800T A2 |
   |  | Atlas 900 A2 PoD |
   |  | Atlas 200T A2 Box16 |
   | Atlas A3 training products | Atlas 800T A3 training server |
   |  | Atlas 900 A3 SuperPoD |
    
   Ascend inference devices include the following models, all of which can be used as the inference environment for large models.
       
   | Product Series | Product Model |
   |-----------------------|----------------------------------|
   | Atlas 800I A2 products | Atlas 800I A2 |

2. Software compatibility

    <a id="table1"></a>

   | PyTorch | TorchNPU | OpPlugin | Python                      | GCC  |
   |---------|------------------------------|----------|-----------------------------|------|
   | 2.7.1   | v2.7.1                       | master   | 3.9, 3.10, 3.11, 3.12, 3.13 | 11.2 |
   | 2.8.0   | v2.8.0                       | master   | 3.9, 3.10, 3.11, 3.12, 3.13 | 13.3 |
   | 2.9.0   | v2.9.0                       | master   | 3.10, 3.11, 3.12, 3.13      | 13.3 |
   | 2.10.0  | v2.10.0                      | master   | 3.10, 3.11, 3.12, 3.13      | 13.3 |
   | 2.11.0  | v2.11.0                      | master   | 3.10, 3.11, 3.12, 3.13      | 13.3 |
   | 2.12.0  | v2.12.0                      | master   | 3.10, 3.11, 3.12, 3.13      | 13.3 |
   | 2.13.0  | master                       | master   | 3.10, 3.11, 3.12, 3.13      | 13.3 |

## Installation Dependencies

You are advised to use the development image provided by TorchNPU for compilation. There are two ways to obtain the image: pull an already built image directly from the Ascend image repository, or build one yourself using the Dockerfile.

- Pull the image directly

    A usable development image is provided for you to compile and build OpPlugin. You can pull it directly from the Ascend image repository: [torch-npu-devel](https://www.hiascend.com/developer/ascendhub/detail/3b0ca76864884546acd07845f6153ee6)

    Taking Atlas A2 training products as an example, the command for pulling the image is as follows:

    ```bash
    docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu-devel:2.13.0-cann9.1.0-910b-manylinux_2_28
    ```

    After the image is pulled, run the following command to start and enter the Docker container, and mount the OpPlugin source code into the container:

    ```bash
    docker run -it -v /{code_path}/op-plugin:/home/op-plugin swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu-devel:2.13.0-cann9.1.0-910b-manylinux_2_28 bash
    ```

- Build the image yourself

    This repository (op-plugin/docker/devel) provides a usable Dockerfile, which can automatically detect the architecture and pull the image. You can read the README file in that directory for more information and build your own development environment according to its instructions.

    ```bash
    cd op-plugin/docker/devel
    export DOCKER_BUILDKIT=1
    docker build -t op-plugin-builder:v1 .
    ```

    Alternatively, use the one-click container creation script builder.sh directly.

    ```bash
    cd op-plugin/docker/devel
    export DOCKER_BUILDKIT=1
    bash builder.sh --cann
    ```

    After the build is complete, run the following command to start and enter the Docker container, and mount the OpPlugin source code into the container:

    ```bash
    docker run -it -v /{code_path}/op-plugin:/home/op-plugin op-plugin-builder:v1 bash
    ```

_{code_path}_ indicates the path of the OpPlugin source code. Replace it according to the actual situation.

> [!NOTE]
>
> - A directly started container can be used only to compile the OpPlugin.
> - To run OpPlugin on the NPU environment in the container, ensure that the driver exists on the host (you can confirm this using the `npu-smi` command) and mount the driver when starting the container. For details, see the README in this directory (op-plugin/docker/devel).
> - In container scenarios, images and source code are obtained from external networks. For network problems related to proxy configuration, see the [official Docker documentation](https://docs.docker.com/engine/cli/proxy/).

In physical machine and virtual machine scenarios, you need to install the system dependencies and the official PyTorch framework yourself. For dependency installation guidance, see [TorchNPU](https://gitcode.com/Ascend/pytorch/tree/master#from-source).
 
## Procedure
 
1. Configure the CANN environment variable script.
 
   ```bash
   source <CANN_install_dir>/<CANN_path>/set_env.sh
   ```
 
   The default path of the environment variable script is `/usr/local/npu/ascend-toolkit/set_env.sh`. The installation path of `ascend-toolkit` depends on the name of the installed CANN software.
 
2. Compile and generate the binary installation package of the plug-in.
 
   Download the branch code of the corresponding OpPlugin version and go to the root directory of the plug-in.

   ```bash
   git clone --branch master https://gitcode.com/ascend/op-plugin.git
   cd op-plugin
   ```

   Perform the compilation and build. The following command uses PyTorch 2.13.0 and Python 3.10 as an example.

   ```bash
   bash ci/build.sh --python=3.10 --pytorch=v2.13.0-26.2.0
   ```

    > [!NOTICE] 
    > For the GCC and Python versions used during compilation, see the constraints in the [software compatibility table](#table1).
    > During compilation, a build folder is created in the root directory of the plug-in, and the source code of the corresponding TorchNPU version is downloaded for collaborative compilation. If the build/pytorch directory exists, the TorchNPU source code is not downloaded again when OpPlugin is compiled. To download the latest TorchNPU source code that OpPlugin depends on, delete the build/pytorch directory.
 
3. After the compilation is complete, install the plug-in TorchNPU package generated in the dist directory. If you install it as a non-root user, add `--user` to the command.
 
   ```bash
   pip3 install --upgrade dist/torch_npu-{torch_npu_version}-{Python_version}-{arch}.whl
   # In actual execution, replace the package name according to the generated whl package. {torch_npu_version} indicates the compiled TorchNPU version, {Python_version} indicates the Python version in use, and {arch} indicates the target architecture.
   # A typical whl package name is similar to: torch_npu-2.13.0rc1-cp310-cp310-linux_aarch64.whl
   ```

## Uninstallation

 Just run the following command to uninstall torch:

 ```bash
 pip uninstall torch_npu
 ```
