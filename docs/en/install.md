# Compiling the Source Code and Installing OpPlugin

## Installation instructions

1. Hardware mapping table
    
    The Ascend training device includes the following models, which can be used as the training environment of the PyTorch model.

    | Product range            | Product Model                              |
    | ------------------------ | ------------------------------------------ |
    | Atlas Training Series    | Atlas 800 training server (model: 9000)    |
    |                          | Atlas 800 training server (model: 9010)    |
    |                          | Atlas 900 PoD (model: 9000)                |
    |                          | Atlas 300T training card (model: 9000)     |
    |                          | Atlas 300T Pro training card (model: 9000) |
    | Atlas A2 Training Series | Atlas 800T A2 training server              |
    |                          | Atlas 900 A2 PoD cluster base unit         |
    |                          | Atlas 200T A2 Box16 Heterogeneous Subrack  |
    | Atlas A3 Training Series | Atlas 800T A3 training server              |
    |                          | Atlas 900 A3 SuperPoD Super Node           |

    The Ascend inference device includes the following models, which can be used as the inference environment for large models.

    | Product range                   | Product Model                  |
    | ------------------------------- | ------------------------------ |
    | Atlas 800I A2 inference product | Atlas 800I A2 inference server |

2. Software mapping table

    <a id="table1"></a>

    | PyTorch | TorchNPU | OpPlugin | Python                      | GCC  |
    | ------- | -------- | -------- | --------------------------- | ---- |
    | 2.7.1   | v2.7.1   | master   | 3.9, 3.10, 3.11, 3.12, 3.13 | 11.2 |
    | 2.8.0   | v2.8.0   | master   | 3.9, 3.10, 3.11, 3.12, 3.13 | 13.3 |
    | 2.9.0   | v2.9.0   | master   | 3.10, 3.11, 3.12, 3.13      | 13.3 |
    | 2.10.0  | v2.10.0  | master   | 3.10, 3.11, 3.12, 3.13      | 13.3 |
    | 2.11.0  | v2.11.0  | master   | 3.10, 3.11, 3.12, 3.13      | 13.3 |
    | 2.12.0  | v2.12.0  | master   | 3.10, 3.11, 3.12, 3.13      | 13.3 |
    | 2.13.0  | master   | master   | 3.10, 3.11, 3.12, 3.13      | 13.3 |

## Installation Dependency

You are advised to use the development image provided by TorchNPU for compilation. The image can be obtained in either of the following ways: directly pull the pre-built image from the Ascend image repository, or build the image by yourself using the Dockerfile.

- Pull the image directly

    A ready-to-use development image is provided for compiling OpPlugin. You can pull it directly from the Ascend image repository: [torch-npu-devel](https://www.hiascend.com/developer/ascendhub/detail/3b0ca76864884546acd07845f6153ee6)

    Taking the Atlas A2 training products as an example, run the following command to pull the image:

    ```bash
    docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu-devel:2.13.0-cann9.1.0-910b-manylinux_2_28
    ```

    After the image is pulled, use the following command to start and access the Docker container and mount the OpPlugin source code to the container:

    ```bash
    docker run -it -v /{code_path}/op-plugin:/home/op-plugin swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu-devel:2.13.0-cann9.1.0-910b-manylinux_2_28 bash
    ```

- Build the image by yourself

    This repository (op-plugin/docker/devel) provides a ready-to-use Dockerfile that can automatically detect the architecture to pull the base image. You can read the README file in that directory for more information and build your own development environment as instructed.

    ```bash
    cd op-plugin/docker/devel
    export DOCKER_BUILDKIT=1
    docker build -t op-plugin-builder:v1 .
    ```

    Alternatively, use the one-click container creation script builder.sh:

    ```bash
    cd op-plugin/docker/devel
    export DOCKER_BUILDKIT=1
    bash builder.sh --cann
    ```

    After the image is built, use the following command to start and access the Docker container and mount the OpPlugin source code to the container:

    ```bash
    docker run -it -v /{code_path}/op-plugin:/home/op-plugin op-plugin-builder:v1 bash
    ```

_{code_path}_ indicates the path of the OpPlugin source code. Replace it based on the actual situation.

> [!NOTE]
>
> - The container started directly can only be used for compiling the OpPlugin plug-in.
> - To run OpPlugin in the NPU environment inside the container, ensure that the driver exists on the host (which can be confirmed by the `npu-smi` command) and mount the driver when starting the container. For details, see the README in the directory (op-plugin/docker/devel).
> - In the container scenario, the image and source code need to be obtained from the external network. For network issues such as proxy configuration, see the [Docker official documentation](https://docs.docker.com/engine/cli/proxy/).

In the physical machine and virtual machine scenarios, you need to install the system dependencies and the official PyTorch framework by yourself. For details about the dependency installation, see the [TorchNPU](https://gitcode.com/Ascend/pytorch/blob/master/README.md#from-source).

## Operation Procedure

1. Script for configuring CANN environment variables.
    
    ```bash
    source <CANN_install_dir>/<CANN_path>/set_env.sh
    ```
    
    The default path of the environment variable script is /usr/local/npu/ascend-toolkit/set_env.sh. The path of ascend-toolkit depends on the name of the installed CANN software.
2. Generate the binary installation package of the plug-in.
    
    Download the branch code of the corresponding OpPlugin version and go to the root directory of the plug-in.
    
    ```bash
    git clone --branch master https://gitcode.com/ascend/op-plugin.git
    cd op-plugin
    ```
    
    Run the following commands to compile and build the plug-in. The following uses PyTorch 2.13.0 and Python 3.10 as an example.
    
    ```bash
    bash ci/build.sh --python=3.10 --pytorch=v2.13.0-26.2.0
    ```
    
    > [!NOTICE]
    >
    > For details about the GCC and Python versions during compilation, see the [Software mapping table](#table1) During the compilation, the build folder is created in the root directory of the plug-in, and the source code of the corresponding TorchNPU version is downloaded for collaborative compilation. If the build/pytorch directory exists, the TorchNPU source code will not be downloaded repeatedly during OpPlugin compilation. To download the latest TorchNPU source code, delete the build/pytorch directory.

3. After the compilation is complete, install the TorchNPU package generated in the dist directory. If a non-root user is used to install the TorchNPU package, add the following information to the end of the command:`--user`.
    
    ```bash
    pip3 install --upgrade dist/torch_npu-{torch_npu_version}-{Python_version}-{arch}.whl
    # Replace it with the name of the generated whl package. {torch_npu_version} indicates the TorchNPU version, {Python_version} indicates the Python version, and {arch} indicates the target architecture.
    # A typical whl package name is similar to torch_npu-2.13.0rc1-cp310-cp310-linux_aarch64.whl.
    ```

## Uninstalling

Run the following command to uninstall the torch:

```bash
pip uninstall torch_npu
```
