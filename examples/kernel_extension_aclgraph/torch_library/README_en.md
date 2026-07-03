# Custom Operator Direct Call and ACLGraph Integration

## Overview

This example project demonstrates how to register custom operators using PyTorch `torch.library`, call kernel functions through the `<<<>>>` kernel launch syntax, and adapt `aclgraph` to use these custom operators. It uses a basic Add operator and an in-place trigonometric operator as examples to demonstrate how to call custom operators within `aclgraph`.

## Supported Products

- Atlas A3 training products/Atlas A3 inference products
- Atlas A2 training products/Atlas A2 inference products

## Directory Structure

```text
├── README.md                   // Example project description
├── setup.py                    // Build configuration file
├── csrc
│   ├── add_custom.asc          // Add operator implementation and custom operator registration
│   └── trig_inplace_custom.asc // In-place trigonometric operator implementation and custom operator registration
├── op_extension
│   ├── __init__.py             // Python initialization file
│   └── _load.py                // Loading module
└── test
    ├── add_aclgraph_test.py    // ACLGraph test case for the Add operator
    └── trig_aclgraph_test.py   // ACLGraph test case for the in-place trigonometric operator            
```

## Operator Description

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

### In-Place Trigonometric Operator

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

## Code Implementation

  - Using the Add operator as an example, this example project defines a namespace named `ascendc_ops` in the `*.asc` file and registers the `ascendc_add` function within the namespace. Within the `ascendc_add` function, the current NPU stream is obtained through `c10_npu::getCurrentNPUStream()`, and the custom kernel function `add_custom` is launched by using the kernel launch syntax `<<<>>>` to execute the operator on the NPU.

    ```c++
      add_custom<<<blockDim, nullptr, aclStream>>>(xGm, yGm, zGm, totalLength);
    ```
  
  - PyTorch provides the `TORCH_LIBRARY_FRAGMENT` macro as the core API for registering custom operators. It is used to create and initialize a custom operator library. After registration, the custom operator can be called on the Python side using `torch.ops.namespace.op_name`. For example:
  
    ```c++
    TORCH_LIBRARY_FRAGMENT(ascendc_ops, m)
    {
        m.def(ascendc_add"(Tensor x, Tensor y) -> Tensor");
    }
    ```
  
  - `TORCH_LIBRARY_IMPL` is used to bind the operator logic to a specific `DispatchKey` (the PyTorch device scheduling identifier). For NPU devices, the operator implementation must be registered to the dedicated `PrivateUse1` dispatch key. For example:
  
    ```c++
    TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m)
    {
        m.impl("ascendc_add", TORCH_FN(ascendc_ops::ascendc_add));
    }
    ```

  - Register the meta function.
  
    Registering a meta function enables the `FakeTensor` pipeline, which is required for features such as `fx` and `torch.compile`. The code sample is as follows:

    ```c++
    TORCH_LIBRARY_IMPL(ascendc_ops, Meta, m)
    {
      m.impl("ascendc_add", &add_impl_meta);
    }
    ```

  - Call `aclgraph`.

    In the [code sample](./test/add_aclgraph_test.py), the generated custom operator library is loaded using `torch.ops.load_library`. It demonstrates three ways to enable `aclgraph`, validating the numerical correctness of the custom operator by comparing the NPU output with the CPU standard addition results.

  1. torch.npu.NPUGraph()
  2. torch.npu.make_graphed_callables
  3. backend="npugraph_ex"

## Build and Execution

Perform the following steps in the root directory of this example project to build and run the operator.

- Environment Setup
  
1. For detailed installation steps for PyTorch and `torch_npu`, refer to the [Ascend Extension for PyTorch Software Installation Guide](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html) corresponding to your version.
       
   This example project requires PyTorch 2.6.0 or later, and support for `backend="npugraph_ex"` requires `torch_npu` 7.3.0 or later.
2. Install the CANN Toolkit package based on the actual environment. This example project requires version 8.5.0 or later. For installation details, see the [CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware).
3. Install the CANN OPS package based on the actual environment. Download the appropriate package based on your hardware model and system architecture by using the [Download Link](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1), and run the following commands:
   
   ```bash
   # Ensure the installer has execute permission
   chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
   # Installation command
   ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run  --install --quiet --install-path=${install_path}
   ```
   
   - `${soc_name}`: NPU model name obtained by removing the prefix "ascend" from `${soc_version}`.
   - `${install_path}`: Installation path, which must be the same as the Toolkit installation path. The default path is `/usr/local/Ascend`.

- Configure environment variables.
  
  Configure the environment variables based on the installation path of the CANN development toolkit on your system.
  
    ```bash
    source ${install_path}/ascend-toolkit/set_env.sh
    ```

- Execute the code sample.

  Modify the `--npu-arch` option in `setup.py` according to the Ascend AI processor architecture by referring to the [Common compilation options](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/BishengCompiler/atlas_bisheng_10_0010.html) table. Then, run the following commands:

  ```bash
  python setup.py bdist_wheel
  pip install dist/*.whl --force-reinstall
  cd test
  python ./add_aclgraph_test.py
  ```

Output similar to the following indicates that the precision comparison is successful:

```bash
Ran * test in **s.
OK
```
