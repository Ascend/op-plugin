# （beta）class c10_npu::NPUStream

## 定义文件

torch_npu\csrc\core\npu\NPUStream.h

## 功能说明

NPUStream是一个NPU流类，实现了NPU流管理的相关功能，是属于NPU设备的线性执行序列。

## 成员函数

- **c10_npu::NPUStream::NPUStream(c10::Stream stream)**

    NPUStream构造函数，从指定流构造NPU流，与c10::cuda::CUDAStream::CUDAStream(c10::Stream  _stream_)相同。

    stream：Stream类型，指定输入流，必须为NPU流。

- **c10_npu::NPUStream::NPUStream(Unchecked, c10::Stream stream)**

    NPUStream构造函数，从指定流构造NPU流，不检测是否为NPU流，与c10::cuda::CUDAStream::CUDAStream(Unchecked, c10::Stream  _stream_)相同。

    stream：Stream类型，指定输入流。

- **c10_npu::NPUStream::\~NPUStream()**

    NPUStream析构函数，与c10::cuda::CUDAStream::\~CUDAStream()相同。

- **bool c10_npu::NPUStream::operator==(const c10_npu::NPUStream& other)**

    NPUStream重载函数，重载==判定，返回true如果转成c10::Stream后二者相等，与bool c10::cuda::CUDAStream::operator==(const c10::cuda::CUDAStream&  _other_)相同。

    other：NPUStream类型，待比较的NPU流。

- **bool c10_npu::NPUStream::operator!=(const c10_npu::NPUStream& other)**

    NPUStream重载函数，重载!=判定，返回true如果转成c10::Stream后二者不相等，与bool c10::cuda::CUDAStream::operator!=(const c10::cuda::CUDAStream&  _other_)相同。

    other：NPUStream类型，待比较的NPU流。

- **c10_npu::NPUStream::operator aclrtStream()**

    NPUStream aclrtStream类型转换，与c10::cuda::CUDAStream::operator cudaStream_t()相同。

- **c10_npu::NPUStream::operator c10::Stream()**

    NPUStream c10::Stream类型转换，与c10::cuda::CUDAStream::operator c10::Stream()相同。

- **c10::DeviceType c10_npu::NPUStream::device_type()**

    NPUStream设备类型获取，返回值类型DeviceType，与c10::DeviceType c10::cuda::CUDAStream::device_type()相同。

- **c10::DeviceIndex c10_npu::NPUStream::device_index()**

    NPUStream设备id获取，返回值类型DeviceIndex，与c10::DeviceIndex c10::cuda::CUDAStream::device_index()相同。

- **c10::Device c10_npu::NPUStream::device()**

    NPUStream设备获取，返回值类型Device，保证是NPU设备，与c10::Device c10::cuda::CUDAStream::device()相同。

- **c10::StreamId c10_npu::NPUStream::id()**

    NPUStream流id获取，返回值类型StreamId，与c10::StreamId c10::cuda::CUDAStream::id()相同。

- **bool c10_npu::NPUStream::query()**

    NPUStream流完成查询，返回值类型bool，true表示所有提交的工作已经完成，与bool c10::cuda::CUDAStream::query()相同。

- **void c10_npu::NPUStream::synchronize()**

    NPUStream流同步，等待直到工作完成，与void c10::cuda::CUDAStream::synchronize()相同。

- **aclrtStream c10_npu::NPUStream::stream()**

    NPUStream aclrtStream流查询，返回值类型aclrtStream，与cudaStream_t c10::cuda::CUDAStream::stream()相同。

- **c10::Stream c10_npu::NPUStream::unwrap()**

    NPUStream Stream流查询，返回值类型Stream，与c10::Stream c10::cuda::CUDAStream::unwrap()相同。

- **struct c10::StreamData3 c10_npu::NPUStream::pack3()**

    NPUStream流压缩，返回值类型StreamData3，与struct c10::StreamData3 c10::cuda::CUDAStream::pack3()相同。

- **c10_npu::NPUStream c10_npu::NPUStream::unpack3(c10::StreamId stream_id, c10::DeviceIndex device_index, c10::DeviceType device_type)**

    NPUStream流解压，从struct StreamData3中解压出流，返回值类型NPUStream，与c10::cuda::CUDAStream c10::cuda::CUDAStream::unpack3(c10::StreamId  _stream_id_, c10::DeviceIndex  _device_index_, c10::DeviceType  _device_type_)相同。

    stream_id：StreamId类型，struct StreamData3中的流id。

    device_index：DeviceIndex类型，struct StreamData3中的设备id。

    device_type：DeviceType类型，struct StreamData3中的设备类型。

- **void c10_npu::NPUStream::setDataPreprocessStream(bool is_data_preprocess_stream)**

    NPUStream is_data_preprocess_stream设置。

    is_data_preprocess_stream：bool类型，true表示将该流设置为数据预处理流。

- **bool c10_npu::NPUStream::isDataPreprocessStream()**

    NPUStream数据预处理流查询，返回值类型bool，返回true表示是数据预处理流。

- **aclrtStream c10_npu::NPUStream::stream(const bool need_empty)**

    NPUStream aclrtStream流查询，返回值类型aclrtStream，允许输入参数need_empty。

    need_empty：bool类型，false表示直接返回当前stream，不用返回空stream。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

