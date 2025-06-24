# c10d_npu::ProcessGroupHCCL::batch_isend_irecv

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term> Atlas A3 训练系列产品</term>             |    √     |
|<term> Atlas A2 训练系列产品</term>   | √   |     

## 功能说明

发送或接收一批tensor，异步处理P2P操作序列中的每一个操作，并返回对应的请求。

## 定义文件

torch_npu\csrc\distributed\ProcessGroupHCCL.hpp

## 函数原型

```
c10::intrusive_ptr<c10d::Work> batch_isend_irecv(std::vector<std::string>& op_type, std::vector<at::Tensor>& tensors, std::vector<uint32_t> remote_rank_list)
```

## 参数说明

- **op_type** (`std::vector<std::string>&`)：必选参数，表示操作序列，isend或irecv。
- **tensors** (`std::vector<at::Tensor>&`)：必选参数，表示用于发送或接收的tensor本身，数量与op_type保持一致。
- **remote_rank_list** (`std::vector<uint32_t>`)：必选参数，表示对端的rank id, 这里指全局的rank id，数量与op_type保持一致。

## 返回值说明

`c10::intrusive_ptr<c10d::Work>`

表示所有的isend和irecv任务，因为是异步操作的，所以将work返回，当进行tensor读取时，需要先执行work.wait()，保证batch_isend_irecv完成。

## 约束说明

无