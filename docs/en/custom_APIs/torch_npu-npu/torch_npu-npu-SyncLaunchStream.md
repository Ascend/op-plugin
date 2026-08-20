# torch_npu.npu.SyncLaunchStream

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas training products</term>                                      |    √     |

## Function

Creates a synchronous delivery `NPUStream`. Tasks delivered on this stream no longer use the `taskqueue` for asynchronous delivery. In cluster scenarios, when a device fails and other devices save checkpoints, this synchronous stream can be used for checkpoint saving.

## Prototype

```python
torch_npu.npu.SyncLaunchStream(device)
```

## Parameters

**`device`** (`Any`): Device identifier, which can be an integer device ID or a string such as `"npu:0"`. The default value is `None`, which corresponds to the device ID bound to the current thread.

## Return Values

The created synchronous `NPUStream` instance. Tasks launched on this stream are not asynchronously delivered through the `taskqueue`.

## Constraints

- Because tasks bypass the `taskqueue`, this stream has lower delivery performance compared to a normal stream. You are advised to create a synchronous `NPUStream` only for checkpoint saving when certain nodes fail during cluster training.
- The synchronous stream pool supports up to four streams. Creating more than four streams causes streams to be reused cyclically from the pool.

## Example

```python
import torch
import torch_npu
s = torch_npu.npu.SyncLaunchStream()
with torch.npu.stream(s):
    tensor1 =torch.randn(4).npu()
    tensor2 = tensor1 + tensor1
    s.synchronize()
```
