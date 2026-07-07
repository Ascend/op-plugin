# Adaptation Development and Usage (Ascend C-based)

This example project demonstrates the complete adaptation development workflow for calling custom Ascend C operators through the `torch_npu` framework by using C++ extensions. It shows how to integrate Ascend C operators into the PyTorch ecosystem to achieve efficient NPU-accelerated computation.

## Operator Adaptation Development

### Prerequisites

Before getting started, ensure that you have completed the installation of the following environments:

1. Install the NPU driver, firmware, and CANN software (including the Toolkit, ops, and NNAL packages) by referring to [CANN Software Installation](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum) (Commercial Edition) or [CANN Software Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum) (Community Edition).
2. Install the PyTorch framework by referring to [Ascend Extension for PyTorch Software Installation Guide](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/installation_guide/installation_description.md).

### Adaptation File Structure

```text
├── build_and_run.sh                // Script for compiling and installing the custom operator wheel package and executing the test case
├── csrc                            // Directory of the C++ code at the operator adaptation layer
│   ├── add_custom.asc              // Adaptation code, ATen IR registration, and binding for the custom Add operator
│   └── trig_inplace_custom.asc     // Adaptation code, ATen IR registration, and binding for the custom Trig operator
├── cpp_extension_acs               // Python-side code for the custom operator package
│   ├── _load.py                    // Loads the .so file
│   └── __init__.py                 // Python initialization file
├── setup.py                        // Compilation file of the wheel package
└── test                            // Directory for test cases
    ├── add_aclgraph_test.py        // Add sample script
    └── trig_aclgraph_test.py       // Trig sample script
```

> [!NOTE]
> 
> The adaptation files above use the Add and Trig operators as examples. Users can adapt their own custom operators. For more information about the Add and Trig operators, see the [Reference](#reference) section.

### Procedure

The following steps use the Add operator as an example.

1. Implement the C++ operator code, adaptation code, custom operator schema registration, and implementation binding in the corresponding `*.asc` file (such as `add_custom.asc`) under the operator adaptation layer C++ directory (`csrc`). A namespace named `cpp_extension_acs` is defined in `add_custom.asc`, where the `ascendc_add` function is registered. Within the `ascendc_add` function, the current NPU stream is obtained through `c10_npu::getCurrentNPUStream()`, and the custom kernel function `add_custom` is launched by using the kernel launch syntax `<<<>>>` to execute the operator on the NPU. PyTorch provides the `TORCH_LIBRARY` macro to define a new namespace and register the schema within the namespace. Note that the namespace name must be unique. The code sample is as follows:
<!--The code sample does not include `const c10::OptionalDeviceGuard device_guard(device_of(Tensor))`. Is this required?-->
    > [!NOTE]
    > 
    > In multi-device scenarios, you must add `const c10::OptionalDeviceGuard device_guard(device_of(Tensor))` to the adaptation code to ensure proper cross-device access. This statement can be omitted in single-device scenarios.

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

2. Add the operator call logic and load the `.so` file in the `__init__.py` and `_load.py` files under the `cpp_extension_acs` directory. The code sample is as follows:

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
        
## Usage Example

After completing the operator adaptation development, you can call the custom operator through C++ extensions.

1. Create the custom operator project and complete the operator development, compilation, and deployment process. For details, see the [CANN Ascend C Operator Development Guide](https://www.hiascend.com/document/detail/en/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html).
2. Download the sample code.

    ```bash
    # Download sample code
    git clone https://gitcode.com/Ascend/op-plugin
    # Go to the code directory
    cd examples/cpp_extension_base
    ```

3. Complete operator adaptation. For details, see [Operator Adaptation Development](#operator-adaptation-development).
4. Run the following command to compile, install, and execute the test script:

    ```bash
    bash build_and_run.sh
    ```

    The following output indicates successful execution:

    ```bash
    Ran xx tests in xx s
    OK
    ```

## Reference

### Add Operator

- Description:

  Adds two input tensors and returns the result. The corresponding operator prototype is as follows:
  
  ```python
  ascendc_add(Tensor x, Tensor y) -> Tensor
  ```

- Specifications
  
  <table>
   <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">add_custom</td></tr>
  <tr><td rowspan="3" align="center">Operator input</td><td align="center">Name</td><td align="center">Shape</td><td align="center">Data Type</td><td align="center">Layout</td></tr>
  <tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">int</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">int</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Operator output</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">int</td><td align="center">ND</td></tr>
  </table>

### Trig Operator

- Description:

  This operator takes `x`, `out_sin`, and `out_cos` as inputs. After the operator is called, `out_sin` is modified in place to store the result of `sin(x)`, `out_cos` is modified in place to store the result of `cos(x)`, and the result of `tan(x)` is returned. The corresponding operator prototype is as follows:
  
  ```python
  ascendc_trig(Tensor x, Tensor(a!) out_sin, Tensor(b!) out_cos) -> Tensor
  ```

- Specifications

  <table>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">trig_inplace_custom</td></tr>
  <tr><td rowspan="4" align="center">Operator input</td><td align="center">Name</td><td align="center">Shape</td><td align="center">Data Type</td><td align="center">Layout</td></tr>
  <tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">out_sin</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">out_cos</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="3" align="center">Operator output</td><td align="center">out_sin</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">out_cos</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">out_tan</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  </table>
