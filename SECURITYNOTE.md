# OpPlugin安全声明

## 系统安全加固
建议用户在系统中配置开启ASLR（级别2 ），又称**全随机地址空间布局随机化**，可参考以下方式进行配置：

    echo 2 > /proc/sys/kernel/randomize_va_space

## 运行用户建议
OpPlugin的运行依赖torch_npu，本章内容请参考[torch_npu仓运行用户建议](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E8%BF%90%E8%A1%8C%E7%94%A8%E6%88%B7%E5%BB%BA%E8%AE%AE)。

## 文件权限控制
1. 用户安装和使用过程需要做好权限控制，建议参考[文件（夹）各场景权限管控推荐最大值](#文件（夹）各场景权限管控推荐最大值)进行设置。如需要保存安装/卸载日志，可在安装/卸载命令后面加上参数--log <FILE>， 注意对<FILE>文件及目录做好权限管控。

2. 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。

##### 文件（夹）各场景权限管控推荐最大值

|   类型                             |   Linux权限参考最大值   |
|----------------------------------- |-----------------------|
|  用户主目录                         |   750（rwxr-x---）     |
|  程序文件(含脚本文件、库文件等)       |   550（r-xr-x---）     |
|  程序文件目录                       |   550（r-xr-x---）     |
|  配置文件                           |   640（rw-r-----）     |
|  配置文件目录                       |   750（rwxr-x---）     |
|  日志文件(记录完毕或者已经归档)       |   440（r--r-----）     |
|  日志文件(正在记录)                  |   640（rw-r-----）    |
|  日志文件目录                       |   750（rwxr-x---）     |
|  Debug文件                         |   640（rw-r-----）      |
|  Debug文件目录                      |   750（rwxr-x---）     |
|  临时文件目录                       |   750（rwxr-x---）     |
|  维护升级文件目录                   |   770（rwxrwx---）      |
|  业务数据文件                       |   640（rw-r-----）      |
|  业务数据文件目录                   |   750（rwxr-x---）      |
|  密钥组件、私钥、证书、密文文件目录   |   700（rwx------）      |
|  密钥组件、私钥、证书、加密密文       |   600（rw-------）     |
|  加解密接口、加解密脚本              |   500（r-x------）      |

## 调试工具声明
OpPlugin的运行依赖torch_npu，本章内容请参考[torch_npu仓调试工具声明](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E8%B0%83%E8%AF%95%E5%B7%A5%E5%85%B7%E5%A3%B0%E6%98%8E)。

## 数据安全声明
OpPlugin的运行依赖torch_npu，本章内容请参考[torch_npu仓数据安全声明](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E6%95%B0%E6%8D%AE%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E)。

## 构建安全声明
OpPlugin的运行依赖torch_npu，本章内容请参考[torch_npu仓构建安全声明](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E6%9E%84%E5%BB%BA%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E)。

## 运行安全声明
OpPlugin的运行依赖torch_npu，本章内容请参考[torch_npu仓运行安全声明](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E8%BF%90%E8%A1%8C%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E)。

## 公网地址声明
在OpPlugin的配置文件和脚本中存在[公网地址](#公网地址)。

##### 公网地址

|   类型   |   开源代码地址   |   文件名   |   公网IP地址/公网URL地址/域名/邮箱地址   |   用途说明   |
|------------------------|-------------------------|-------------------------|-------------------------|-------------------------|
|   开发引入  |   不涉及   |   ci\build.sh   |   https://gitee.com/ascend/pytorch.git   |   编译脚本根据torch_npu仓库地址拉取代码进行编译   |
|   开发引入  |   不涉及   |   ci\exec_ut.sh   |   https://gitee.com/ascend/pytorch.git   |   UT脚本根据torch_npu仓库地址下拉取代码进行UT测试   |
| 开源代码引入 |pytorch\aten\src\ATen\native\TensorCompare.cpp  | op_plugin\ops\opapi\IsInKernelNpuOpApi.cpp | https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/arraysetops.py#L575| 算法实现借鉴numpy的源码地址|

## 公开接口声明
OpPlugin的运行依赖torch_npu，不提供公开接口。

## 通信安全加固
OpPlugin的运行依赖torch_npu，本章内容请参考[torch_npu仓通信安全加固](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E5%AE%89%E5%85%A8%E5%8A%A0%E5%9B%BA)。

## 通信矩阵
OpPlugin的运行依赖torch_npu，本章内容请参考[torch_npu仓通信矩阵](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E7%9F%A9%E9%98%B5)。