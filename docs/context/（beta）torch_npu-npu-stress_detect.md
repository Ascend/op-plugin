# （beta）torch\_npu.npu.stress\_detect

>**说明：**<br>
>此接口为beta接口，属于实验性接口，部分场景下可能出现异常，请谨慎使用此接口。

## 函数原型

```
torch_npu.npu.stress_detect()
```

## 功能说明

提供硬件精度在线检测接口，供模型调用。主要通过StressDetect接口实现，该接口会对硬件做压力测试检测是否存在静默精度问题。

## 输出说明

返回值为int，代表错误类型，含义如下所示：

- 0：在线硬件精度检测通过。

- 1：在线硬件精度检测用例执行失败。

- 2：在线硬件精度检测不通过，硬件故障。

- runtime error：电压恢复失败。

## 约束说明

1.  硬件精度在线检测的使用需要修改用户的模型训练脚本，建议在训练开始前、结束后、两个step之间调用，同时需要预留10G大小的内存供压测接口使用。
2.  硬件精度在线检测用例仅支持<term>Atlas A2 训练系列产品</term>、<term>Atlas A3 训练系列产品</term>，不支持在同一节点运行多个训练作业，同时调压功能不支持算力切分场景；不建议使用多线程运行在线精度检测用例。

## 支持的型号

-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>

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
            print(f"Epoch {epoch + 1}/{num_epochs}: Stress detection skipped (called too frequently).")
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

