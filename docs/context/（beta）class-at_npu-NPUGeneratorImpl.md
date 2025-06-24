# （beta）class at_npu::NPUGeneratorImpl

## 定义文件

torch_npu\csrc\aten\NPUGeneratorImpl.h

## 功能说明

NPUGeneratorImpl是一个随机数生成器类，实现了NPU设备随机数的相关功能，可用于众多依赖随机数的方法。

## 成员函数

- **at_npu::NPUGeneratorImpl::NPUGeneratorImpl(c10::DeviceIndex device_index = -1)**

    NPUGeneratorImpl构造函数，指定NPU设备id构造生成器，与at::CUDAGeneratorImpl::CUDAGeneratorImpl(c10::DeviceIndex  _device_index_)相同。

    device_index：DeviceIndex类型，指定npu设备id。

- **std::shared_ptr at_npu::NPUGeneratorImpl::clone()**

    NPUGeneratorImpl拷贝函数，返回值类型shared_ptr，返回NPUGeneratorImpl拷贝，与std::shared_ptr at::CUDAGeneratorImpl::clone()相同。

- **void at_npu::NPUGeneratorImpl::set_current_seed(uint64_t seed)**

    NPUGeneratorImpl随机种子设置，设置当前的随机种子，与at::CUDAGeneratorImpl::set_current_seed(uint64_t  _seed_)相同。

    seed：uint64_t类型，待设置的随机种子。

- **void at_npu::NPUGeneratorImpl::set_offset(uint64_t offset)**

    NPUGeneratorImpl offset设置，与at::CUDAGeneratorImpl::set_offset(uint64_t  _offset_)相同。

    offset：uint64_t类型，待设置的offset。

- **uint64_t at_npu::NPUGeneratorImpl::current_seed()**

    NPUGeneratorImpl随机种子获取，返回值类型uint64_t，返回当前的随机种子，与at::CUDAGeneratorImpl::current_seed()相同。

- **uint64_t at_npu::NPUGeneratorImpl::get_offset()**

    NPUGeneratorImpl offset获取，返回值类型uint64_t，返回offset，与at::CUDAGeneratorImpl::get_offset()相同。

- **uint64_t at_npu::NPUGeneratorImpl::seed()**

    NPUGeneratorImpl随机种子更新，返回值类型uint64_t，生成并返回新的随机种子，与at::CUDAGeneratorImpl::seed()相同。

- **void at_npu::NPUGeneratorImpl::set_state(const c10::TensorImpl& new_state)**

    NPUGeneratorImpl状态设置，设置指定状态，与void at::CUDAGeneratorImpl::set_state(const c10::TensorImpl&  _new_state_)相同。

    new_state：TensorImpl类型，待设置的状态，需要通过at::detail::check_rng_state检测。

- **c10::intrusive_ptr c10::TensorImpl at_npu::NPUGeneratorImpl::get_state()**

    NPUGeneratorImpl状态获取，返回值类型intrusive_ptr，返回生成器状态，与c10::intrusive_ptrc10::TensorImpl at::CUDAGeneratorImpl::get_state()相同。

- **void at_npu::NPUGeneratorImpl::set_philox_offset_per_thread(uint64_t offset)**

    NPUGeneratorImpl设置每条线程的philox offset，用于curandStatePhilox4_32_10，与void at::CUDAGeneratorImpl::set_philox_offset_per_thread(uint64_t  _offset_)相同。

    offset：uint64_t类型，待设置的philox offset。

- **uint64_t at_npu::NPUGeneratorImpl::philox_offset_per_thread()**

    NPUGeneratorImpl获取每条线程的philox offset，返回值类型uint64_t，与uint64_t at::CUDAGeneratorImpl::philox_offset_per_thread()相同。

- **void at_npu::NPUGeneratorImpl::capture_prologue(int64_t\*offset_extragraph)**

    NPUGeneratorImpl设置offset_extragraph，用于NpuGraph来预留图捕获区域，支持图捕获，与void at::CUDAGeneratorImpl::capture_prologue(int64_t\*  _seed_extragraph_, int64_t\*  _offset_extragraph_)相同。

    offset_extragraph：int64_t\*类型，待设置的offset_extragraph。

- **uint64_t at_npu::NPUGeneratorImpl::capture_epilogue()**

    NPUGeneratorImpl结束图捕获，返回值类型uint64_t，关闭图捕获并返回预留区域大小offset_extragraph，与uint64_t at::CUDAGeneratorImpl::capture_epilogue()相同。

- **at_npu::PhiloxNpuState at_npu::NPUGeneratorImpl::philox_npu_state(uint64_t increment)**

    NPUGeneratorImpl philox npu state捕获，返回值类型PhiloxNpuState，与at::PhiloxCudaState at::CUDAGeneratorImpl::philox_cuda_state(uint64_t  _increment_)相同。

    increment：uint64_t类型，philox offset增量。

- **std::pair<uint64_t, uint64_t> at_npu::NPUGeneratorImpl::philox_engine_inputs(uint64_t increment)**

    NPUGeneratorImpl philox_engine输入获取，返回值类型pair<uint64_t, uint64_t>，包含随机种子和philox offset值，与std::pair<uint64_t, uint64_t> at::CUDAGeneratorImpl::philox_engine_inputs(uint64_t  _increment_)相同。

    increment：uint64_t类型，philox offset增量。

- **c10::DeviceType at_npu::NPUGeneratorImpl::device_type()**

    NPUGeneratorImpl设备类型获取，返回值类型DeviceType，与c10::DeviceType at::CUDAGeneratorImpl::device_type()相同。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

