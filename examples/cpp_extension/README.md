## 目录结构介绍
```
├── examples
|   ├── cpp_extension
|   │   ├── op_extension          // python脚本, 初始化模块
|   │   ├── csrc                  // C++目录
|   │   │   ├── kernel            // 算子kernel实现    
|   │   |   ├── host              // 算子注册torch
|   │   |   |   ├── tiling        // 算子tiling实现
|   |   ├── test                  // 测试用例目录
|   |   ├── CMakeLists.txt        // Cmake文件
|   |   ├── setup.py              // setup文件
|   |   ├── README.md             // 模块使用说明  
```
## 新增自定义KERNEL_LAUNCH算子
支持Ascend C实现自定义算子Kernel，并集成在Pytorch框架，通过Pytorch的API调用实现算子。
### kernel实现
  本小节主要介绍如何实现Kernel算子，本样例基于Ascend C进行开发算子。如何使用Ascend C实现算子kernel，可以参考昇腾社区文档[昇腾Ascend C](https://www.hiascend.com/ascend-c)。


  新增一个算子实现，需要在./cpp_extension/csrc/kernel目录下添加Kernel算子文件。Ascend C编写的算子，是否需要workspace具有不同编译选项。样例中提供了两种算子的实现，add和matmul_leakyrelu(需要workspace)。用户可按需新增算子实现，并将对应kernel文件在CMakeLists.txt中添加到编译。具体编译选项说明和使用方法可参考昇腾社区文档[昇腾Ascend C](https://www.hiascend.com/ascend-c)。

  add kernel在CMakeLists.txt中添加样例
  ```
ascendc_library(no_workspace_kernel STATIC
    csrc/kernel/add_custom.cpp
)
  ```
matmul_leakyrelu在CMakeLists.txt中添加样例
  ```
ascendc_library(workspace_kernel STATIC
    csrc/kernel/matmul_leakyrelu_custom.cpp
)
ascendc_compile_definitions(workspace_kernel PRIVATE
  -DHAVE_WORKSPACE
  -DHAVE_TILING
)
  ```



### 将Kernel实现与Pytorch集成
  本小节主要介绍如何封装实现的kernel算子，以及注册为pythorch的API。
  代码实现在./cpp_extension/csrc/host 目录下，其中子目录tiling存放算子的tiling函数。

#### Aten IR定义
pytorch通过`TORCH_LIBRARY`宏提供了一个简单方法，将C++算子实现绑定到python。python侧可以通过`torch.ops.namespace.op_name`方式进行调用。这种方式包括Aten IR的定义和注册。我们通过`TORCH_LIBRARY`实现注册，如果相同namespace在不同文件中有注册，需要使用`TORCH_LIBRARY_FRAGMENT`。
例如将自定义`my_add`函数注册在`npu`命名空间下：
  ```c++
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("my_add(Tensor x, Tensor y) -> Tensor");
}
  ```
  通过此注册，python侧可通过`torch.ops.npu.my_add`调用自定义的API。
#### Aten IR实现

步骤1：按需包含头文件。需要注意的是，需要包含对应的核函数调用接口声明所在的头文件，alcrtlaunch_{kernel_name}.h（该头文件为Ascend C工程框架自动生成），kernel_name为算
子核函数的名称

步骤2：算子适配，根据Aten IR定义适配算子，包括按需实现输出Tensor申请，workspace申请，调用kernel算子等。
TORCH_NPU的算子下发和执行是异步的，通过TASKQUEUE实现，
样例中，我们通过`EXEC_KERNEL_CMD`宏封装了算子的`ACLRT_LAUNCH_KERNEL`方法，将算子执行入队到TORCH_NPU的TASKQUEUE。样例如下：

  ```c++
#include "utils.h"
#include "aclrtlaunch_add_custom.h"
at::Tensor run_add_custom(const at::Tensor &x, const at::Tensor &y)
{
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 8;
    uint32_t totalLength = 1;
    for (uint32_t size : x.sizes()) {
        totalLength *= size;
    }
    EXEC_KERNEL_CMD(add_custom, blockDim, x, y, z, totalLength);
    return z;
}
  ```
#### Aten IR注册
通过pytorch提供的`TORCH_LIBRARY_IMPL`注册算子实现，运行在NPU设备上需要注册在`PrivateUse1`这个dispatchkey上。
样例如下：
  ```c++
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("my_add", TORCH_FN(run_add_custom));
}
  ```

上述主要介绍了一个自定义算子kernel如何集成在pytorch必备流程。具体实现可参考我们提供的两个样例Add和Matmul_Leakyrelu算子，包含了如何实现带有workspace、tiling的样例。

我们在host目录下的utils.h文件中提供了一些工具函数，包括算子下发，tensor拷贝等，如果用户想要实现一些更高阶的能力，可按需实现。


## 运行自定义的算子
  运行依赖torch、torch_npu和CANN。具体安装步骤参考[torch_npu文档](https://gitee.com/ascend/pytorch#%E5%AE%89%E8%A3%85)
  运行流程：
  1. 设置编译的AI处理器型号
  
  修改CmakeLists.txt内的SOC_VERSION为所需产品型号。对应代码位置如下：
  ```bash
  set(SOC_VERSION "Ascendxxxyy" CACHE STRING "system on chip type")
  ```
  需将`Ascendxxxyy`修改为对应产品型号。

  AI处理器的型号<soc_version>请通过如下方式获取：

      在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，获取Chip Name信息。实际
      配置值为AscendChip Name，例如Chip Name取值为xxxyy，实际配置值为Ascendxxxyy

  2. 运行setup脚本，编译生成whl包。
    ```bash
    python setup.py bdist_wheel
    ```
  我们的编译工程通过setuptools已为用户封装好如何编译算子kernel和集成到pytorch，如果需要更多的编译配置，可按需更改CmakeLists.txt文件。

  3. 安装whl包
    ```bash
    cd dist
    pip install *.whl
    ```
  
  4. 运行样例
    ```bash
    cd test
    python test.py
    ``` 
