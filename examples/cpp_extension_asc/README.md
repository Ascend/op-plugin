# 适配开发及调用（AscendC）

基于C++ extensions方式，通过torch_npu框架调用AscendC自定义算子的完整适配开发流程。本样例展示了如何将AscendC算子集成到PyTorch生态中，实现高效的NPU加速计算。

## 算子适配开发

### 前提条件

在开始之前，请确保您已完成以下环境的安装。

1. 请参考《[CANN 快速安装](https://www.hiascend.com/cann/download)》安装昇腾NPU驱动和CANN软件（包含Toolkit、ops和NNAL包），并配置环境变量。
2. 请参考《[Ascend Extension for PyTorch 软件安装指南](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/zh/installation_guide/installation_description.md)》完成PyTorch框架的安装。

### 适配文件结构

```text
├── build_and_run.sh                // 自定义算子wheel包编译安装并执行用例的脚本
├── csrc                            // 算子适配层c++代码目录
│   ├── add_custom.asc              // 自定义add算子适配代码、ATen IR注册以及绑定
│   └── trig_inplace_custom.asc     // 自定义trig算子适配代码、ATen IR注册以及绑定
├── cpp_extension_acs               // 自定义算子包python侧代码
│   ├── _load.py                    // 调用so
│   └── __init__.py                 // python初始化文件
├── setup.py                        // wheel包编译文件
└── test                            // 测试用例目录
    ├── add_aclgraph_test.py        // add用例脚本
    └── trig_aclgraph_test.py       // trig用例脚本
```

> [!NOTE]
> 
> 以上适配文件仅以add算子和trig算子为例，用户可自行适配自定义算子，更多add算子和trig算子信息可参考[算子详情](#参考信息)。

### 操作步骤

以下步骤均以add算子为例。

1. 在算子适配层C++代码目录（csrc）中的*.asc文件（如add_custom.asc）完成C++侧算子代码、适配代码、注册自定义算子schema及绑定具体实现。在add_custom.asc中定义了一个名为cpp_extension_acs的命名空间，并在其中注册了ascendc_add函数。在ascendc_add函数中通过`c10_npu::getCurrentNPUStream()`函数获取当前NPU上的流，并通过内核调用符<<<>>>调用自定义的Kernel函数add_custom，在NPU上执行算子。PyTorch提供TORCH_LIBRARY宏来定义新的命名空间，并在该命名空间里注册schema。注意命名空间的名字必须是唯一的。具体示例如下：
<!-- 代码中没有const c10::OptionalDeviceGuard device_guard(device_of(Tensor))，是否需要增加 -->
    > [!NOTE]
    > 
    > 多卡场景必须在适配代码中加`const c10::OptionalDeviceGuard device_guard(device_of(Tensor))`保障跨device访问，单卡场景可不加此代码。

    ```cpp
    constexpr uint32_t BUFFER_NUM = 2;  //tensor num for each queue
    class KernelAdd {
    public:
        __aicore__ inline KernelAdd() {}
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
        {
            this->blockLength = totalLength / AscendC::GetBlockNum();
            this->tileNum = 8;
            this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
            xGm.SetGlobalBuffer((__gm__ int32_t *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
            yGm.SetGlobalBuffer((__gm__ int32_t *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
            zGm.SetGlobalBuffer((__gm__ int32_t *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
            pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(int32_t));
            pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(int32_t));
            pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(int32_t));
        }
        __aicore__ inline void Process()
        {
            int32_t loopCount = this->tileNum * BUFFER_NUM;
            for (int32_t i = 0; i < loopCount; i++) {
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        }

    private:
        __aicore__ inline void CopyIn(int32_t progress)
        {
            AscendC::LocalTensor<int32_t> xLocal = inQueueX.AllocTensor<int32_t>();
            AscendC::LocalTensor<int32_t> yLocal = inQueueY.AllocTensor<int32_t>();
            AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
            AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
            inQueueX.EnQue(xLocal);
            inQueueY.EnQue(yLocal);
        }
        __aicore__ inline void Compute(int32_t progress)
        {
            AscendC::LocalTensor<int32_t> xLocal = inQueueX.DeQue<int32_t>();
            AscendC::LocalTensor<int32_t> yLocal = inQueueY.DeQue<int32_t>();
            AscendC::LocalTensor<int32_t> zLocal = outQueueZ.AllocTensor<int32_t>();
            AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
            outQueueZ.EnQue<int32_t>(zLocal);
            inQueueX.FreeTensor(xLocal);
            inQueueY.FreeTensor(yLocal);
        }
        __aicore__ inline void CopyOut(int32_t progress)
        {
            AscendC::LocalTensor<int32_t> zLocal = outQueueZ.DeQue<int32_t>();
            AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
            outQueueZ.FreeTensor(zLocal);
        }

    private:
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
        AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
        AscendC::GlobalTensor<int32_t> xGm;
        AscendC::GlobalTensor<int32_t> yGm;
        AscendC::GlobalTensor<int32_t> zGm;
        uint32_t blockLength;
        uint32_t tileNum;
        uint32_t tileLength;
    };

    __global__ __vector__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        KernelAdd op;
        op.Init(x, y, z, totalLength);
        op.Process();
    }

    namespace cpp_extension_acs {
    at::Tensor ascendc_add(const at::Tensor &x, const at::Tensor &y)
    {
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
        at::Tensor z = at::empty_like(x);
        uint32_t blockDim = 8;
        uint32_t totalLength = 1;
        for (uint32_t size : x.sizes()) {
            totalLength *= size;
        }
        // Launch the custom kernel use <<<>>>
        add_custom<<<blockDim, nullptr, acl_stream>>>((uint8_t *)(x.mutable_data_ptr()), (uint8_t *)(y.mutable_data_ptr()),
                                                        (uint8_t *)(z.mutable_data_ptr()), totalLength);
        return z;
    }

    }  // namespace cpp_extension_acs

    at::Tensor add_impl_meta(const at::Tensor& x, const at::Tensor& y)
    {
        return at::empty_like(x);
    }

    // Define a new operator
    TORCH_LIBRARY_FRAGMENT(cpp_extension_acs, m)
    {
        m.def("ascendc_add(Tensor x, Tensor y) -> Tensor");
    }

    // Register implementation for the "PrivateUse1" backend
    TORCH_LIBRARY_IMPL(cpp_extension_acs, PrivateUse1, m)
    {
        m.impl("ascendc_add", TORCH_FN(cpp_extension_acs::ascendc_add));
    }

    // Define a simple model using the custom operation
    TORCH_LIBRARY_IMPL(cpp_extension_acs, Meta, m)
    {
        m.impl("ascendc_add", &add_impl_meta);
    }

    ```

2. 在`cpp_extension_acs`目录下的`__init__.py`及`_load.py`文件中，添加ops调用及读取so文件，具体示例如下：

    ```Python
    # __init__.py
    from ._load import _load_opextension_so
    _load_opextension_so()

    # _load.py
    import pathlib
    import torch
    # Load the custom operator library
    def _load_opextension_so():
        so_dir = pathlib.Path(__file__).parents[0]
        so_files = list(so_dir.glob('custom_ops_lib*.so'))

        if not so_files:
            raise FileNotFoundError(f"not find custom_ops_lib*.so in {so_dir}")

        atb_so_path = str(so_files[0])
        torch.ops.load_library(atb_so_path)
    ```
        
## 调用样例

完成了算子适配开发后，即可实现C++ extensions的方式调用自定义算子。

1. 完成自定义算子工程创建、算子开发及编译部署流程，具体可参考《[CANN Ascend C算子开发指南](https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html)》。
2. 下载示例代码。

    ```bash
    # 下载样例代码
    git clone https://gitcode.com/Ascend/op-plugin
    # 进入代码目录
    cd examples/cpp_extension_base
    ```

3. 完成算子适配，具体可参考[算子适配开发](#算子适配开发)。
4. 执行如下命令，完成编译、安装，并运行测试脚本。

    ```bash
    bash build_and_run.sh
    ```

    得到结果如下即为执行成功。

    ```bash
    Ran xx tests in xx s
    OK
    ```

## 参考信息

### Add算子

- 算子功能：

  Add算子实现了两个数据相加，返回相加结果的功能。对应的算子原型为：
  
  ```python
  ascendc_add(Tensor x, Tensor y) -> Tensor
  ```

- 算子规格：
  
  <table>
   <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">int</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">int</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">int</td><td align="center">ND</td></tr>
  </table>

### trig算子

- 算子功能：

  该算子入参为x, out_sin ,out_cos, 算子调用后，out_sin会被原地修改为sin(x)计算结果，out_cos会被原地修改为cos(x)计算结果，返回值tan(x)计算结果。对应的算子原型为：
  
  ```python
  ascendc_trig(Tensor x, Tensor(a!) out_sin, Tensor(b!) out_cos) -> Tensor
  ```

- 算子规格：

  <table>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">trig_inplace_custom</td></tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">out_sin</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">out_cos</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="3" align="center">算子输出</td><td align="center">out_sin</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">out_cos</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">out_tan</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  </table>
