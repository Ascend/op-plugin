# (beta) torch_npu.npu.stress_detect

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

Provides online hardware precision detection APIs for use by models. This API is implemented through the `StressDetect` interface, which performs a hardware stress test to detect silent precision issues.

## Prototype

```python
torch_npu.npu.stress_detect(detect_type="aic")
```

## Parameters

**`detect_type`** (`str`): Optional. Detection type. Valid values are `aic` or `hccs`, representing online hardware precision detection and online HCCS link precision detection, respectively. Configuring any other value blocks execution and returns `1`. The default value is `aic`.

> [!NOTE]  
> When `detect_type` is set to `hccs`, sub-communicators for all local cards are first created based on the global communication domain, followed by an HCCS link stress test on these sub-communicators.

## Return Values

- An integer error code representing execution status. Valid values are:

    - `0`: Online precision detection passed.

    - `1`: Online precision detection test case failed to execute.

    - `2`: Online precision detection failed due to a hardware fault.

- The following error indicates a voltage recovery failure, which requires manual voltage restoration by referring to [LINK](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/maintenref/troubleshooting/troubleshooting_0505.html) or a system reboot.

    ```shell
    Stress detect error. Error code is 574007. Error message is Voltage recovery failed.
    ```

## Constraints

- Online precision detection requires modifying the model training script. You are advised to call this API before training starts, after training ends, or between two steps. Reserve 10 GB of memory for the stress test interface.
- HCCS link online detection (`detect_type="hccs"`) can be performed only after the global communication domain has been initialized.
- Online precision detection is not supported when running multiple training jobs on a single node. The voltage scaling feature is not supported in computing power partitioning scenarios.
- Running online precision detection in multi-threaded environments is not recommended.

## Example

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
