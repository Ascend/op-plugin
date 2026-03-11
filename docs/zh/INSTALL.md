## 安装OpPlugin

### 发布包安装
暂未正式发布

### 源码编译安装

#### 安装依赖

安装时需要安装系统依赖及官方PyTorch框架，建议使用torch_npu提供的docker镜像进行编译，依赖安装和镜像使用指导可参考[Ascend Extension for Pytorch](https://gitcode.com/Ascend/pytorch/tree/v2.7.1-7.3.0#%E4%BD%BF%E7%94%A8%E6%BA%90%E4%BB%A3%E7%A0%81%E8%BF%9B%E8%A1%8C%E5%AE%89%E8%A3%85)。

#### 操作步骤

1. 配置CANN环境变量脚本。

   ```
   source <CANN软件安装目录>/<CANN软件路径>/set_env.sh
   ```

   环境变量脚本的默认路径一般为：/usr/local/Ascend/ascend-toolkit/set_env.sh，其中ascend-toolkit路径取决于安装的CANN软件名称。

2. 编译生成插件的二进制安装包。

   下载对应OpPlugin版本分支代码，进入插件根目录，以v2.7.1为例。
   ```
   git clone --branch 7.3.0 https://gitcode.com/ascend/op-plugin.git
   cd op-plugin
   ```
   执行编译构建，当前支持torch_npu 2.6.0/2.7.1/2.8.0/2.9.0版本，下述命令中`v2.7.1-7.3.0`表示匹配OpPlugin仓7.3.0版本的PyTorchv2.7.1的分支名。
   ```
   bash ci/build.sh --python=3.8 --pytorch=v2.7.1-7.3.0
   ```
   >**须知：**<br>
   >请注意，编译时gcc版本遵循如下约束：
   >   - ARM架构下推荐使用gcc 10.2版本
   >   - X86架构下推荐使用gcc 9.3.1; PyTorch 2.6.0及之后版本的编译推荐使用gcc 11.2.1
   

   | 参数      | 取值范围                                                           | 说明                  | 缺省值    | 备注                                |
   |---------|----------------------------------------------------------------|---------------------|--------|-----------------------------------|
   | PyTorch | v2.6.0-7.3.0, v2.7.1-7.3.0, v2.8.0-7.3.0, v2.9.0-7.3.0 | 指定编译过程中使用的PyTorch版本对应的分支名 | master | 需要显示指定分支名，避免使用缺省值master      |
   | Python  | 3.8, 3.9, 3.10, 3.11                                           | 指定编译过程中使用的Python版本  | 3.8    | 编译基于PyTorch2.6及后续版本时，不支持Python3.8 |

   | PyTorch      | Python               | GCC                |
   |--------------|----------------------|--------------------|
   | v2.6.0       | 3.9, 3.10, 3.11      | ARM: 11.2 x86: 9.3 |
   | v2.7.1       | 3.9, 3.10, 3.11      | 11.2               |
   | v2.8.0       | 3.9, 3.10, 3.11      | 13.3               |
   | v2.9.0       | 3.10, 3.11           | 13.3               |
   | master(2.10) | 3.10, 3.11           | 13.3               |

   >编译过程中，会在插件根目录新建build文件夹，并下载torch_npu对应版本的源码，协同编译。 若build/pytorch目录存在，则编译op-plugin时，不再重复下载torch_npu源码。如需下载所依赖的最新torch_npu源码，删除build/pytorch目录即可。

3. 完成编译后，安装dist目录下生成的插件torch\_npu包，如果使用非root用户安装，需要在命令后加`--user`。

   ```
   pip3 install --upgrade dist/torch_npu-{torch_npu_version}-{Python_version}-{arch}.whl
   # 实际执行时需要根据生成的whl包名称进行替换，其中{torch_npu_version}表示编译的torch_npu版本，{Python_version} 为所使用的 Python 版本，{arch} 则代表目标架构。
   # 典型的whl包名类似：torch_npu-2.7.1.post13-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
   ```