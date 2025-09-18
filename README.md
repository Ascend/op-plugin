# OpPlugin

## 简介

本项目开发了Ascend Extension for Pytorch（torch_npu）算子插件，为使用PyTorch框架的开发者提供便捷的NPU算子库调用能力。
OpPlugin算子插件的编译、使用依赖昇腾Ascend Extension for PyTorch。因此，在编译OpPlugin之前，需要了解、安装昇腾PyTorch。使用手册可参考昇腾社区[Ascend Extension for Pytorch](https://gitcode.com/ascend/pytorch/blob/master/README.zh.md)。

## 版本配套表
op-plugin仓旨在为**torch_npu**提供运行所需要的算子适配文件，两个仓的对应关系如下：
| op-plugin分支 | 对应Ascend Extension for PyTorch版本 |
| ------------- | :----------------------------------: |
| master        |     主线版本，如v2.1.0等             |
| 7.1.0        |     7.1.0版本，如v2.1.0-7.1.0等       |
| 7.0.0        |     7.0.0版本，如v2.1.0-7.0.0等       |
| 6.0.0        |     6.0.0版本，如v2.1.0-6.0.0等       |
| 6.0.rc3       |   6.0.rc3版本，如v2.1.0-6.0.rc3等    |
| 6.0.rc2       |   6.0.rc2版本，如v2.1.0-6.0.rc2等    |
| 6.0.rc1       |   6.0.rc1版本，如v2.1.0-6.0.rc1等    |
| 5.0.0         |     5.0.0版本，如v2.1.0-5.0.0等      |
| 5.0.rc3       |   5.0.rc3版本，如v2.1.0-5.0.rc3等    |

## 安装OpPlugin

### 发布包安装
暂未正式发布

### 源码编译安装

#### 安装依赖

安装时需要安装系统依赖及官方PyTorch框架，建议使用torch_npu提供的docker镜像进行编译，依赖安装和镜像使用指导可参考[Ascend Extension for Pytorch](https://gitcode.com/ascend/pytorch/blob/master/README.zh.md#%E4%BD%BF%E7%94%A8%E6%BA%90%E7%A0%81%E7%BC%96%E8%AF%91%E5%AE%89%E8%A3%85pytorch%E6%A1%86%E6%9E%B6)。

#### 操作步骤

1. 配置CANN环境变量脚本。

   ```
   source <CANN软件安装目录>/<CANN软件路径>/set_env.sh
   ```

   环境变量脚本的默认路径一般为：/usr/local/Ascend/ascend-toolkit/set_env.sh，其中ascend-toolkit路径取决于安装的CANN软件名称。

2. 编译生成插件的二进制安装包。

   下载对应OpPlugin版本分支代码，进入插件根目录，以v2.1为例。
   ```
   git clone --branch master https://gitcode.com/ascend/op-plugin.git
   cd op-plugin
   ```
   执行编译构建，当前支持torch_npu 2.1/2.3/2.4/2.5/2.6/2.7/2.8/master版本。
   ```
   bash ci/build.sh --python=3.8 --pytorch=v2.1.0
   ```
   >**须知：**<br>
   >请注意，编译时gcc版本遵循如下约束：
   >   - ARM架构下推荐使用gcc 10.2版本
   >   - X86架构下推荐使用gcc 9.3.1; PyTorch 2.6.0及之后版本的编译推荐使用gcc 11.2.1
   

   | 参数      | 取值范围                                                           | 说明                  | 缺省值    | 备注                                |
   |---------|----------------------------------------------------------------|---------------------|--------|-----------------------------------|
   | PyTorch | v2.1.0, v2.3.1, v2.4.0, v2.5.1, v2.6.0, v2.7.1, v2.8.0, master | 指定编译过程中使用的PyTorch版本 | master | master对应的PyTorch版本为2.9            |
   | Python  | 3.8, 3.9, 3.10, 3.11                                           | 指定编译过程中使用的Python版本  | 3.8    | 编译基于PyTorch2.5及后续版本时，不支持Python3.8 |

   | PyTorch      | Python               | GCC                |
   |--------------|----------------------|--------------------|
   | v2.1.0       | 3.8, 3.9, 3.10, 3.11 | ARM: 10.2 x86: 9.3 |
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
   # 典型的whl包名类似：torch_npu-2.1.0.post13-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
   ```


## 硬件配套

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


## 生命周期
op-plugin仓依赖**torch_npu**运行，生命周期请参考**torch_npu**中的[PyTorch版本维护策略](https://gitcode.com/ascend/pytorch/blob/master/README.zh.md#pytorch%E7%89%88%E6%9C%AC%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5)。


## 安全声明

[OpPlugin安全声明](SECURITYNOTE.md)
