# （beta）class at_npu::native::OpCommand

## 定义文件

torch_npu\csrc\framework\OpCommand.h

## 功能说明

OpCommand是一个封装下层算子调用的类，实现了NPU设备下层算子调用的相关功能。

## 成员函数

- **at_npu::native::OpCommand::OpCommand()**

    OpCommand构造函数，创建一个OpCommand。

- **at_npu::native::OpCommand::\~OpCommand()**

    OpCommand析构函数。

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Name(const string& name)**

    OpCommand待执行算子名称，返回值类型OpCommand。

    name：string类型，待执行的算子名称。

- **at_npu::native::OpCommand& at_npu::native::OpCommand::SetCustomHandler(PROC_FUNC func)**

    OpCommand设置自定义处理方法，返回值类型OpCommand。

    func：PROC_FUNC类型，待设置的自定义处理方法。

- **at_npu::native::OpCommand& at_npu::native::OpCommand::DynamicInputReg(DynamicInputRegFunc func, DyNumAndIndex num_and_index)**

    OpCommand动态输入方法注册，返回值类型OpCommand。

    func：DynamicInputRegFunc类型，待注册的动态输入方法。

    num_and_index：DyNumAndIndex类型，待注册的id。

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Expect(UnifiedResult unified_result)**

    OpCommand设置预期结果形式，返回值类型OpCommand。

    unified_result：UnifiedResult类型，包含预期结果的类型、形状和是否已定义。

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Input()**

    OpCommand空输入，返回值类型OpCommand。

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Input(const at::Tensor& input, const string& descName = "", const c10::optional& sensitive_format = c10::nullopt, const string& realData = "")**

    OpCommand输入Tensor，要求连续，返回值类型OpCommand。

    input：Tensor类型，输入tensor，要求是连续的。

    descName：string类型，名称描述。

    sensitive_format：aclFormat类型，optional，特定format要求。

    realData：string类型，实际数据类型。

- **at_npu::native::OpCommand& at_npu::native::OpCommand::InputWithoutContiguous(const at::Tensor& input, const string& descName = "", const string& realData = "")**

    OpCommand输入Tensor，不要求连续，返回值类型OpCommand。

    input：Tensor类型，输入tensor。

    descName：string类型，名称描述。

    realData：string类型，实际数据类型。

- **template at_npu::native::OpCommand& at_npu::native::OpCommand::Input(const c10::ArrayRef& dimListRef, at::IntArrayRef realShape, at::ScalarType toType, CompileType compileType = CompileType::MEMORY_HOST_COMPILE_DEPENDENT, const string& realDtype = "", const string& descName = "")**

    OpCommand输入数组，通常输入在cpu端，会在实现中进行h2d，返回值类型OpCommand。

    dimListRef：ArrayRef类型，输入数组。

    realShape：IntArrayRef类型，输入形状。

    toType：ScalarType类型，目标类型。

    compileType：CompileType类型，编译类型。

    realDtype：string类型，实际dtype。

    descName：string类型，名称描述。

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Input(const c10::IntArrayRef& dimListRef, at::ScalarType toType = at::kLong, CompileType compileType = CompileType::MEMORY_HOST_COMPILE_DEPENDENT, const string& realDtype = "", const string& descName = "")**

    OpCommand输入整型数组，通常输入在cpu端，会在实现中进行h2d，返回值类型OpCommand。

    dimListRef：IntArrayRef类型，输入整型数组。

    toType：ScalarType类型，目标类型。

    compileType：CompileType类型，编译类型。

    realDtype：string类型，实际dtype。

    descName：string类型，名称描述。

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Input(const c10::ArrayRefdimListRef, at::IntArrayRef realShape, at::ScalarType toType = at::kDouble, CompileType compileType = CompileType::MEMORY_HOST_COMPILE_DEPENDENT, const string& realDtype = "")**

    OpCommand输入浮点型数组，通常输入在cpu端，会在实现中进行h2d，返回值类型OpCommand。

    dimListRef：ArrayRef类型，输入浮点型数组。

    realShape：IntArrayRef类型，输入形状。

    toType：ScalarType类型，目标类型。

    compileType：CompileType类型，编译类型。

    realDtype：string类型，实际dtype。

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Input(const c10::Scalar& input, const at::ScalarType type, CompileType compileType = CompileType::MEMORY_HOST_COMPILE_INDEPENDENT);**

    OpCommand输入标量，会在实现中进行h2d，返回值类型OpCommand。

    input：Scalar类型引用，输入标量。

    type：ScalarType类型，目标类型。

    compileType：CompileType类型，编译类型。

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Inputs(const at::TensorList& inputs)**

    OpCommand输入tensor list，返回值类型OpCommand。

    inputs：TensorList类型引用，输入的tensor list。

- **at_npu::native::OpCommand& at_npu::native::OpCommand::InputScalarToNPUTensor(const c10::Scalar& input, const at::ScalarType type)**

    OpCommand输入标量，返回值类型OpCommand。

    input：Scalar类型引用，输入标量。

    type：ScalarType类型，目标类型。

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Output(at::Tensor& output, const string& descName = "", const c10::optional& sensitive_format = c10::nullopt, const string& realType = "")**

    OpCommand输出tensor，返回值类型OpCommand。

    output：Tensor类型引用，输出tensor。

    descName：string类型，名称描述。

    sensitive_format：aclFormat类型，optional，特定format要求。

    realData：string类型，实际数据类型。

- **template at_npu::native::OpCommand& at_npu::native::OpCommand::Attr(const string& name, dataType value)**

    OpCommand设置属性，返回值类型OpCommand。

    name：string类型常量引用，属性名称。

    value：dataType类型，属性值。

- **template at_npu::native::OpCommand& at_npu::native::OpCommand::Attr(const string& name, dataType value, bool cond)**

    OpCommand根据条件设置属性，返回值类型OpCommand。

    name：string类型常量引用，属性名称。

    value：dataType类型，属性值。

    cond：bool类型，用于判定的条件，若为false则不设置。

- **void at_npu::native::OpCommand::Run()**

    OpCommand算子执行。

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Sync(c10::SmallVector<int64_t, N\>& sync_index)**

    OpCommand设置同步index，返回值类型OpCommand。

    sync_index：SmallVector<int64_t, N\>类型引用，需要同步的index。

- **at_npu::native::OpCommand& at_npu::native::OpCommand::Sync()**

    OpCommand同步，等待直到流完成。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>


