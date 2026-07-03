# Adaptation Development and Usage

## Directory Structure

```text
├── examples
|   ├── cpp_extension
|   │   ├── csrc
|   │   │   ├── add_custom.asc          # Add operator implementation
|   │   │   ├── trig_inplace_custom.asc # In-place trigonometric operator implementation
|   │   │   └── pybind11.asc            # pybind binding for custom operators
|   |   ├── op_extension/
|   |   │   ├── __init__.py             # Extension loading logic
|   |   │   └── ops/
|   |   │       └── __init__.py         # Python API definition and extension loading logic
|   |   ├── test
|   │   |   └── test.py                 # Test script
|   |   ├── setup.py                    # Build configuration
|   |   └── README.md                   # Documentation
```

## Adding a New Custom Operator

This example project uses the `Add` operator to demonstrate how to implement a custom operator kernel using Ascend C in PyTorch. It also shows how to call the implemented operator through a Python API.

### Kernel Implementation

  This section describes how to implement the kernel operator. This example project is developed based on Ascend C. For information about how to implement an operator kernel using Ascend C, refer to the Ascend Community documentation: [Ascend C](https://www.hiascend.com/ascend-c).

  Create a file named `add_custom.asc` under the `./csrc/` directory, which serves as the implementation file for the custom `Add` kernel. A kernel function named `run_ascendc_add` is implemented in this example project.

### Kernel Implementation and Integration

  This subsection describes how to encapsulate the implemented kernel operator and bind it to a Python API.
  The code is located under the `./csrc/` directory.

#### Encapsulating the Python Module

The `pybind11.asc` file uses the `pybind11` library to encapsulate C++ code into a Python module, which can be imported and used on the Python side by using an `import` statement. For example:
  
  ```c++
  PYBIND11_MODULE(custom_ops, m)
  {
      m.def("custom_add", &ascendc_ops::run_ascendc_add, "");
  }
  ```

  Through this binding, the Python side can call the custom API through `op_extension.ops.custom_add`.

#### ATen IR Implementation

Operators are adapted according to the ATen IR definition.
Operator dispatch and execution in `torch_npu` are asynchronous, implemented through a task queue mechanism.
In this project, the `at_npu::native::OpCommand::RunOpApiV2` method is used to enqueue operator execution into the `torch_npu` task queue. The code sample is as follows:

  ```c++
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
namespace ascendc_ops {
at::Tensor run_ascendc_add(const at::Tensor &x, const at::Tensor &y)
{
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 8;
    uint32_t totalLength = 1;
    for (uint32_t size : x.sizes()) {
        totalLength *= size;
    }
    // Launch the custom kernel use <<<>>>
    auto acl_call = [=]() -> int{
        add_custom<<<blockDim, nullptr, acl_stream>>>((uint8_t *)(x.mutable_data_ptr()), (uint8_t *)(y.mutable_data_ptr()),
                                                    (uint8_t *)(z.mutable_data_ptr()), totalLength);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_add", acl_call);

    return z;
}

}  // namespace ascendc_ops
  ```

The preceding section describes the essential workflow required to integrate a custom operator kernel.

Finally, by creating the `ops` path and defining the Python interface, you can call the custom operator through `module_name.ops.custom_add`. The test code sample is as follows:

  ```c++
import torch
x = torch.randint(low=1, high=100, size=length, device='cpu', dtype=torch.int)
y = torch.randint(low=1, high=100, size=length, device='cpu', dtype=torch.int)

x_npu = x.npu()
y_npu = y.npu()
output = op_extension.ops.custom_add(x_npu, y_npu)
  ```

## Running the Custom Operator

  Execution depends on `torch`, `torch_npu`, and CANN. For details about the installation procedure, see the [torch_npu Documentation](https://gitcode.com/Ascend/pytorch#Installation).
  The execution workflow is as follows:

  1. Run the `setup.py` script to build and generate the wheel package.

      ```bash
      python setup.py bdist_wheel
      ```

     The build system has already encapsulated the process of compiling the operator kernel and integrating it into PyTorch using `setuptools` for users.

  2. Install the wheel package.

      ```bash
      cd dist
      pip install *.whl
      ```
  
  3. Run the example project.

      ```bash
      cd test
      python test.py
      ``` 

## Frequently Asked Questions (FAQ)
 
### 1. Compilation Error: "bisheng Command Not Found"

**Cause**: The `bisheng` compiler is not installed on the system, or environment variables are not correctly configured.  
**Solution**:

- Ensure that the CANN toolkit is correctly installed.
- Run `source /usr/local/Ascend/ascend-toolkit/set_env.sh` to configure environment variables.
- Verify whether the `bisheng` compiler is available by running `bisheng --version`.
 
### 2. Runtime Error: "ModuleNotFoundError: No module named 'op_extension'"

**Cause**: The custom extension package is not installed correctly.  
**Solution**:

- Ensure the wheel package has been successfully built and installed.
- Check whether the installation directory is included in the module search path of Python.
- Try reinstalling the wheel package through `pip install --force-reinstall *.whl`.

## Precautions
 
### 1. Data Type Support

- The current implementation only supports the `int32` data type.
- To support other data types (such as `float32` or `float16`), you must modify the kernel implementation.
- When modifying the implementation, pay attention to the byte size and alignment requirements for data types.
 
### 2. Hardware Compatibility

- This example project is developed based on Ascend NPUs and supports only Ascend hardware.
- Different Ascend NPU models may require adjustments to compilation parameters.
- You must specify the correct NPU architecture during compilation, such as `--npu-arch=dav-2201`.
 
### 3. Version Compatibility

- Ensure that PyTorch, `torch_npu`, and CANN versions are mutually compatible.
- Version mismatch may cause compilation or runtime errors.
- Refer to the [torch_npu Documentation](https://gitcode.com/Ascend/pytorch#Installation) for compatible version information.
