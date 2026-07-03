# (beta) class at_npu::native::OpCommand

## Definition File

torch_npu\csrc\framework\OpCommand.h

## Function

Acts as a class encapsulating low-level operator calls and implements underlying operator execution on NPU devices.

## Member Functions

- **at_npu::native::OpCommand::OpCommand()**

    `OpCommand` constructor, which creates an `OpCommand` instance.

- **at_npu::native::OpCommand::\~OpCommand()**

    `OpCommand` destructor.

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Name(const string& name)**

    Name of the operator to be executed. The return value type is `OpCommand&`.

    **`name`** (`string`): Required. Name of the operator to be executed.

- **at_npu::native::OpCommand& at_npu::native::OpCommand::SetCustomHandler(PROC_FUNC func)**

    Sets a custom handling method for `OpCommand`. The return value type is `OpCommand`.

    **func`** (`PROC_FUNC`): Custom handling method to be set.

- **at_npu::native::OpCommand& at_npu::native::OpCommand::DynamicInputReg(DynamicInputRegFunc func, DyNumAndIndex num_and_index)**

    Registers a dynamic input method for `OpCommand`. The return value type is `OpCommand`.

    **`func`** (`DynamicInputRegFunc`): Dynamic input method to be registered.

    **`num_and_index`** (`DyNumAndIndex`): Identifier to be registered.

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Expect(UnifiedResult unified_result)**

    Sets the expected result format for `OpCommand`. The return value type is `OpCommand`.

    **`unified_result`** (`UnifiedResult`): Includes the type, shape, or definition state of the expected result.

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Input()**

    An empty input for `OpCommand`. The return value type is `OpCommand`.

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Input(const at::Tensor& input, const string& descName = "", const c10::optional& sensitive_format = c10::nullopt, const string& realData = "")**

    Tensor input for `OpCommand`, which must be contiguous. The return value type is `OpCommand`.

    **`input`** (`Tensor`): Input tensor, which must be contiguous.

    **`descName`** (`string`): Name description.

    **`sensitive_format`** (`aclFormat`): Specific format requirements.

    **`realData`** (`string`): Actual data type.

- **at_npu::native::OpCommand& at_npu::native::OpCommand::InputWithoutContiguous(const at::Tensor& input, const string& descName = "", const string& realData = "")**

    Tensor input for `OpCommand`. Non-contiguous tensors are supported. The return value type is `OpCommand`.

    **`input`** (`Tensor`): Input tensor.

    **`descName`** (`string`): Name description.

    **`realData`** (`string`): Actual data type.

- **template at_npu::native::OpCommand& at_npu::native::OpCommand::Input(const c10::ArrayRef& dimListRef, at::IntArrayRef realShape, at::ScalarType toType, CompileType compileType = CompileType::MEMORY_HOST_COMPILE_DEPENDENT, const string& realDtype = "", const string& descName = "")**

    Array input for `OpCommand`, typically provided on the CPU side, which is copied through H2D during execution. The return value type is `OpCommand`.

    **`dimListRef`** (`ArrayRef`): Input array.

    **`realShape`** (`IntArrayRef`): Input shape dimensions.

    **`toType`** (`ScalarType`): Target data type.

    **`compileType`** (`CompileType`): Compilation type.

    **`realDtype`** (`string`): Actual data type.

    **`descName`** (`string`): Name description.

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Input(const c10::IntArrayRef& dimListRef, at::ScalarType toType = at::kLong, CompileType compileType = CompileType::MEMORY_HOST_COMPILE_DEPENDENT, const string& realDtype = "", const string& descName = "")**

    Integer array input for `OpCommand`, typically provided on the CPU side, which is copied through H2D during execution. The return value type is `OpCommand`.

    **`dimListRef`** (`IntArrayRef`): Input integer array.

    **`toType`** (`ScalarType`): Target data type.

    **`compileType`** (`CompileType`): Compilation type.

    **`realDtype`** (`string`): Actual data type.

    **`descName`** (`string`): Name description.

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Input(const c10::ArrayRef& dimListRef, at::IntArrayRef realShape, at::ScalarType toType = at::kDouble, CompileType compileType = CompileType::MEMORY_HOST_COMPILE_DEPENDENT, const string& realDtype = "")**

    Floating-point array input for `OpCommand`, typically provided on the CPU side, which is copied through H2D during execution. The return value type is `OpCommand`.

    **`dimListRef`** (`ArrayRef`): Input floating-point array.

    **`realShape`** (`IntArrayRef`): Input shape dimensions.

    **`toType`** (`ScalarType`): Target data type.

    **`compileType`** (`CompileType`): Compilation type.

    **`realDtype`** (`string`): Actual data type.

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Input(const c10::Scalar& input, const at::ScalarType type, CompileType compileType = CompileType::MEMORY_HOST_COMPILE_INDEPENDENT);**

    Scalar input for `OpCommand`, which is copied via H2D during execution. The return value type is `OpCommand`.

    **`input`** (`Scalar`): Input scalar reference.

    **`type`** (`ScalarType`): Target data type.

    **`compileType`** (`CompileType`): Compilation type.

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Inputs(const at::TensorList& inputs)**

    Tensor list input for `OpCommand`. The return value type is `OpCommand`.

    **`inputs`** (`TensorList`): Input tensor list reference.

- **at_npu::native::OpCommand& at_npu::native::OpCommand::InputScalarToNPUTensor(const c10::Scalar& input, const at::ScalarType type)**

    Scalar input for `OpCommand`. The return value type is `OpCommand`.

    **`input`** (`Scalar`): Input scalar reference.

    **`type`** (`ScalarType`): Target data type.

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Output(at::Tensor& output, const string& descName = "", const c10::optional& sensitive_format = c10::nullopt, const string& realType = "")**

    Output tensor for `OpCommand`. The return value type is `OpCommand`.

    **`output`** (`Tensor`): Output tensor reference.

    **`descName`** (`string`): Name description.

    **`sensitive_format`** (`aclFormat`): Specific format requirements.

    **`realType`** (`string`): Actual data type.

- **template at_npu::native::OpCommand& at_npu::native::OpCommand::Attr(const string& name, dataType value)**

    Sets an attribute for `OpCommand`. The return value type is `OpCommand`.

    **`name`** (`string`): Attribute name constant reference.

    **`value`** (`dataType`): Attribute value.

- **template at_npu::native::OpCommand& at_npu::native::OpCommand::Attr(const string& name, dataType value, bool cond)**

    Conditionally sets an attribute for `OpCommand`. The return value type is `OpCommand`.

    **`name`** (`string`): Attribute name constant reference.

    **`value`** (`dataType`): Attribute value.

    **`cond`** (`bool`): Condition used to determine whether the attribute is set. If `False`, the attribute is not set.

- **void at_npu::native::OpCommand::Run()**

    Executes the `OpCommand` operator.

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Sync(c10::SmallVector<int64_t, N\>& sync_index)**

    Sets synchronization indices for `OpCommand`. The return value type is `OpCommand`.

    **`sync_index`** (`SmallVector<int64_t, N>`): Reference to the indices to be synchronized.

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Sync()**

    Synchronizes `OpCommand` and blocks execution until the execution stream completes.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
