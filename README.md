# AscendPyTorch OpPlugin

### 简介

本项目开发了NPU PyTorch算子插件，为使用PyTorch框架的开发者提供便捷的NPU算子库调用能力。
OP-Plugin算子插件的编译、使用依赖昇腾PyTorch Adapter。因此，在编译op\_plugin之前，需要了解、安装昇腾PyTorch。使用手册可参考昇腾社区[pytorch](https://gitee.com/ascend/pytorch/blob/master/README.zh.md)。

### 编译、安装op_plugin

#### 发布包安装
暂未正式发布

#### 源码安装

**安装依赖**

安装时需要安装系统依赖及官方PyTorch框架。安装指导可参考昇腾pytorch社区[安装依赖](https://gitee.com/ascend/pytorch/blob/master/README.zh.md#%E4%BD%BF%E7%94%A8%E6%BA%90%E7%A0%81%E7%BC%96%E8%AF%91%E5%AE%89%E8%A3%85pytorch%E6%A1%86%E6%9E%B6)。

1. 配置CANN环境变量脚本。

   ```
   source <CANN软件安装目录>/<CANN软件路径>/set_env.sh
   ```

   环境变量脚本的默认路径一般为：/usr/local/Ascend/ascend-toolkit/set_env.sh，其中ascend-toolkit路径取决于安装的CANN软件名称。

2. 编译生成插件的二进制安装包。

   下载对应op_plugin版本分支代码，进入插件根目录，以master为例
   ```
   git clone https://gitee.com/ascend/op-plugin.git
   cd op-plugin
   ```
   执行编译构建，当前支持昇腾pytorch 1.11/2.0/2.1/2.2(master)版本
   ```
   bash ci/build.sh --python=3.8 --pytorch=v2.1.0
   ```

   | 参数      | 取值范围                            | 说明                  | 缺省值    | 备注                                |
   |---------|---------------------------------|---------------------|--------|-----------------------------------|
   | pytorch | v1.11.0, v2.0.1, v2.1.0, master | 指定编译过程中使用的pytorch版本 | master | master对应的pytorch版本为2.2            |
   | python  | 3.7, 3.8, 3.9, 3.10             | 指定编译过程中使用的python版本  | 3.8    | 仅pytorch版本为1.11时才支持指定python版本为3.7 |

   >编译过程中，会在插件根目录新建build文件夹，并下载昇腾pytorch对应版本的源码，协同编译。 若build/pytorch目录存在，则编译op-plugin时，不再重复下载昇腾pytorch源码。如需下载所依赖的最新昇腾pytorch源码，删除build/pytorch目录即可。

3. 完成编译后，安装dist目录下生成的插件torch\_npu包，如果使用非root用户安装，需要在命令后加**--user**。

   ```
   pip3 install --upgrade dist/torch_npu-1.11.0-cp37-cp37m-linux_aarch64.whl
   # 若用户在x86架构下安装插件，请替换为对应的whl包。
   ```
