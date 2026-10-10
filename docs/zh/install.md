# 源码编译安装OpPlugin

## 安装说明

1. 硬件配套表
 
   昇腾训练设备包含以下型号，都可作为PyTorch模型的训练环境。
       
   | 产品系列               | 产品型号                         |
   |-----------------------|----------------------------------|
   | Atlas 训练系列产品     | Atlas 800 训练服务器（型号：9000） |
   |                       | Atlas 800 训练服务器（型号：9010） |
   |                       | Atlas 900 PoD（型号：9000）       |
   |                       | Atlas 300T 训练卡（型号：9000）    |
   |                       | Atlas 300T Pro 训练卡（型号：9000）|
   | Atlas A2 训练系列产品  | Atlas 800T A2 训练服务器          |
   |                       | Atlas 900 A2 PoD 集群基础单元     |
   |                       | Atlas 200T A2 Box16 异构子框      |
   | Atlas A3 训练系列产品  | Atlas 800T A3 训练服务器          |
   |                       | Atlas 900 A3 SuperPoD 超节点     |
    
   昇腾推理设备包含以下型号，都可作为大模型的推理环境。
       
   | 产品系列               | 产品型号                         |
   |-----------------------|----------------------------------|
   | Atlas 800I A2推理产品  | Atlas 800I A2 推理服务器          |

2. 软件配套表

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

## 安装依赖

建议使用TorchNPU提供的开发镜像进行编译。镜像有两种获取方式：直接从昇腾镜像仓库拉取已构建好的镜像，或使用Dockerfile自行构建。

- 直接拉取镜像

    我们已提供了可用的开发镜像，以供您编译构建OpPlugin。您可以从昇腾镜像仓库直接拉取：[torch-npu-devel](https://www.hiascend.com/developer/ascendhub/detail/3b0ca76864884546acd07845f6153ee6)

    以Atlas A2 训练系列产品为例，拉取镜像的命令为：

    ```bash
    docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu-devel:2.13.0-cann9.1.0-910b-manylinux_2_28
    ```

    拉取完成后，使用以下命令启动并进入Docker容器，并将OpPlugin源代码挂载至容器内：

    ```bash
    docker run -it -v /{code_path}/op-plugin:/home/op-plugin swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu-devel:2.13.0-cann9.1.0-910b-manylinux_2_28 bash
    ```

- 自构建镜像

    本仓库（op-plugin/docker/devel）提供了可用的Dockerfile，可以自动检测架构来拉取镜像。你可以阅览该目录下的README文件，获取更多信息，并根据其指导构建自己的开发环境。

    ```bash
    cd op-plugin/docker/devel
    export DOCKER_BUILDKIT=1
    docker build -t op-plugin-builder:v1 .
    ```

    或直接使用一键式创建容器脚本builder.sh。

    ```bash
    cd op-plugin/docker/devel
    export DOCKER_BUILDKIT=1
    bash builder.sh --cann
    ```

    构建完成后，使用以下命令启动并进入Docker容器，并将OpPlugin源代码挂载至容器内：

    ```bash
    docker run -it -v /{code_path}/op-plugin:/home/op-plugin op-plugin-builder:v1 bash
    ```

_{code_path}_ 表示OpPlugin源代码路径，请根据实际情况进行替换。

> [!NOTE]
>
> - 直接启动的容器仅可用于编译OpPlugin插件。
> - 如需在容器内的NPU环境上运行OpPlugin，请确保宿主机已存在驱动（driver）（可通过 `npu-smi` 命令确认），并在启动容器时挂载驱动。具体操作可参考该目录（op-plugin/docker/devel）下README。
> - 容器场景下涉及从外部网络获取镜像及源码，代理配置等相关网络问题请参考[Docker官方文档](https://docs.docker.com/engine/cli/proxy/)。

物理机及虚拟机场景下，需自行安装系统依赖及官方PyTorch框架，依赖安装指导可参考[TorchNPU](https://gitcode.com/Ascend/pytorch/tree/master#%E6%BA%90%E7%A0%81%E7%BC%96%E8%AF%91%E5%AE%89%E8%A3%85)。
 
## 操作步骤
 
1. 配置CANN环境变量脚本。
 
   ```bash
   source <CANN软件安装目录>/<CANN软件路径>/set_env.sh
   ```
 
   环境变量脚本的默认路径一般为：/usr/local/npu/ascend-toolkit/set_env.sh，其中ascend-toolkit路径取决于安装的CANN软件名称。
 
2. 编译生成插件的二进制安装包。
 
   下载对应OpPlugin版本分支代码，进入插件根目录。

   ```bash
   git clone --branch master https://gitcode.com/ascend/op-plugin.git
   cd op-plugin
   ```

   执行编译构建，以下命令以PyTorch版本2.13.0、Python 3.10为例。

   ```bash
   bash ci/build.sh --python=3.10 --pytorch=v2.13.0-26.2.0
   ```

    > [!NOTICE] 
    > 编译时GCC和Python版本请参考[软件配套表](#table1)中约束。
    > 编译过程中，会在插件根目录新建build文件夹，并下载TorchNPU对应版本的源码，协同编译。 若build/pytorch目录存在，则编译OpPlugin时，不再重复下载TorchNPU源码。如需下载所依赖的最新TorchNPU源码，删除build/pytorch目录即可。
 
3. 完成编译后，安装dist目录下生成的插件TorchNPU包，如果使用非root用户安装，需要在命令后加`--user`。
 
   ```bash
   pip3 install --upgrade dist/torch_npu-{torch_npu_version}-{Python_version}-{arch}.whl
   # 实际执行时需要根据生成的whl包名称进行替换，其中{torch_npu_version}表示编译的TorchNPU版本，{Python_version} 为所使用的 Python 版本，{arch} 则代表目标架构。
   # 典型的whl包名类似：torch_npu-2.13.0rc1-cp310-cp310-linux_aarch64.whl
   ```

## 卸载

 只需执行以下命令卸载 torch 即可：

 ```bash
 pip uninstall torch_npu
 ```
