# （beta）torch\_npu.npu.stress\_detect

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |


## 功能说明

提供精度在线检测接口，供模型调用。主要通过`StressDetect`接口实现，该接口会对硬件做压力测试检测是否存在静默精度问题。

## 函数原型

```
torch_npu.npu.stress_detect(detect_type="aic")
```

## 参数说明

**detect_type** (`str`)：可选参数，可支持配置为aic或hccs，分别表示硬件在线精度检测和HCCS链路在线精度检测。配置其他值时直接返回1（表示执行失败），默认值为aic。

> [!NOTE]  
> 当`detect_type`配置为hccs时，首先基于全局通信域创建本机所有卡的子通信域，然后对该子通信域进行HCCS链路压测。


## 返回值说明

- 接口返回值为`int`，代表错误类型，含义如下所示：

    - 0：在线精度检测通过。

    - 1：在线精度检测用例执行失败。

    - 2：在线精度检测不通过，硬件故障。

- 若报如下异常，则表示电压恢复失败，需参见[LINK](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/troubleshooting/troubleshooting_0097.html)手动恢复电压或reboot。
    ```
    Stress detect error. Error code is 574007. Error message is Voltage recovery failed.
    ```
## 约束说明

- 精度在线检测的使用需要修改用户的模型训练脚本，建议在训练开始前、结束后以及两个step之间调用，同时需要预留10G大小的内存供压测接口使用。
- HCCS链路在线检测（`detect_type="hccs"`）需要初始化全局通信域后才能进行调用。
- 精度在线检测用例，不支持在同一节点运行多个训练作业场景下使用，同时调压功能不支持算力切分场景。
- 不建议使用多线程运行在线精度检测用例。


## 调用示例

```python
import torch
import torch_npu

# Custom exception for stress detection failure
class StressDetectionException(Exception):
    def __init__(self, error_code):
        super().__init__(f"Stress detection failed with error code: {error_code}")

# Simple example of model training
def train_model(model, dataloader, optimizer, loss_fn, num_epochs):
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to("npu"), labels.to("npu")

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Call hardware stress detection after each epoch
        stress_detect_result = torch_npu.npu.stress_detect()
        if stress_detect_result == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}: Stress detection passed.")
        elif stress_detect_result == 1:
            print(f"Epoch {epoch + 1}/{num_epochs}: Stress detection failed.")
        else:
            # Raise an exception for any other non-zero result
            raise StressDetectionException(stress_detect_result)

        print(f"Epoch {epoch + 1} Loss: {running_loss/len(dataloader)}")

    print("Training complete.")

# Define a simple model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Sample dataloader
dataloader = [ (torch.randn(32, 10), torch.randn(32, 1)) for _ in range(100) ]

# Create model and move it to Ascend device
model = SimpleModel().to("npu")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Train the model and call stress detection
try:
    train_model(model, dataloader, optimizer, loss_fn, num_epochs=10)
except StressDetectionException as e:
    print(f"Training halted due to: {e}")
    # do something
```

