# （beta）struct c10_npu::NPUEvent

## 定义文件

torch_npu\csrc\core\npu\NPUEvent.h

## 功能说明

NPUEvent是一个事件类，实现了NPU设备事件管理的相关功能，可用于监视设备的进度、精确测量计时以及同步NPU流。

## 成员函数

- **c10_npu::NPUEvent::NPUEvent()**

    NPUEvent空构造函数，与at::cuda::CUDAEvent::CUDAEvent()相同。

- **c10_npu::NPUEvent::NPUEvent(unsigned int flags)**

    NPUEvent带flags的构造函数，与at::cuda::CUDAEvent(unsigned int  _flags_)相同。

    flags：unsigned int类型，指定构造的事件类型。

- **c10_npu::NPUEvent::\~NPUEvent()**

    NPUEvent析构函数，与at::cuda::CUDAEvent::\~CUDAEvent()相同。

- **c10_npu::NPUEvent::NPUEvent(c10_npu::NPUEvent&& other)**

    NPUEvent移动构造函数，与at::cuda::CUDAEvent::CUDAEvent(at::cuda::CUDAEvent&&  _other_)相同。

    other：NPUEvent类型，用于移动构造新的NPUEvent对象。

- **c10_npu::NPUEvent::operator aclrtEvent()**

    NPUEvent aclrtEvent类型转换，与at::cuda::CUDAEvent::operator cudaEvent_t()相同。

- **c10::optional\<at::Device> c10_npu::NPUEvent::device()**

    NPUEvent设备类型获取，与c10::optional\<at::Device> at::cuda::CUDAEvent::device()相同。

- **bool c10_npu::NPUEvent::isCreated()**

    NPUEvent创建询问，返回值类型bool，true表示已创建，与bool at::cuda::CUDAEvent::isCreated()相同。

- **c10::DeviceIndex c10_npu::NPUEvent::device_index()**

    NPUEvent设备id获取，与c10::DeviceIndex at::cuda::CUDAEvent::device_index()相同。

- **aclrtEvent c10_npu::NPUEvent::event()**

    NPUEvent acl事件查询，返回值类型aclrtEvent，与cudaEvent_t at::cuda::CUDAEvent::event()相同。

- **bool c10_npu::NPUEvent::query()**

    NPUEvent事件完成查询，返回值类型bool，true表示所有提交的工作已经完成，与bool at::cuda::CUDAEvent::query()相同。

- **void c10_npu::NPUEvent::record()**

    NPUEvent事件记录，记录当前流的事件，与void at::cuda::CUDAEvent::record()相同。

- **void c10_npu::NPUEvent::record(const c10_npu::NPUStream& stream)**

    NPUEvent事件记录，记录指定流的事件，与void at::cuda::CUDAEvent::record(const c10::CUDA::CUDAStream&  _stream_)相同。

    stream：NPUStream类型，指定记录事件的流。

- **void c10_npu::NPUEvent::recordOnce(const c10_npu::NPUStream& stream)**

    NPUEvent事件记录，给定指定流，若未记录过事件则记录一次，与void at::cuda::CUDAEvent::recordOnce(const c10::CUDA::CUDAStream&  _stream_)相同。

    stream：NPUStream类型，指定记录事件的流。

- **void c10_npu::NPUEvent::block(const c10_npu::NPUStream& stream)**

    NPUEvent事件阻塞，阻塞指定流的事件，与void at::cuda::CUDAEvent::block(const c10::CUDA::CUDAStream&  _stream_)相同。

    stream：NPUStream类型，指定阻塞事件的流。

- **float c10_npu::NPUEvent::elapsed_time(const c10_npu::NPUEvent& other)**

    NPUEvent事件阻塞，返回值类型float，返回记录事件到当前事件经过的时间，单位为ms，与float at::cuda::CUDAEvent::elapsed_time(const at::cuda::CUDAEvent&  _other_)相同。

    other：NPUStream类型，指定计算时间的终点事件。

- **void c10_npu::NPUEvent::synchronize()**

    NPUEvent事件同步，等待直到事件完成，与void at::cuda::CUDAEvent::synchronize()相同。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

