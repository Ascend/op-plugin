# torch_npu.npu.mstx.annotate

## Supported Products

| Product                                                      | Supported |
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 training products</term>                        |    √     |
| <term>Atlas A3 inference products</term>                        |    √     |
| <term>Atlas A2 training products</term>                        |    √     |
| <term>Atlas A2 inference products</term> |    √     |
| <term>Atlas inference products</term>                           |    √     |
| <term>Atlas training products</term>                           |    √     |

## Function

Performs API-level instrumentation that allows users to select APIs or functions for collecting execution duration.

## Prototype

```python
torch_npu.npu.mstx.annotate(message: str = '', stream=None, domain: str = 'default')
```

## Parameters

- **`message`** (`str`): String carrying information for the instrumentation point.

  When this API is called using a `with` statement, this parameter is required. When this API is called as a decorator, this parameter is optional, and the function name is used as the `message` by default.

  The length of the `message` string must not exceed 255 bytes in msPTI scenarios.

- **`stream`** (`torch_npu.npu.Stream`): Optional. Stream used to execute the instrumentation task. The default value is `None`.

  - When set to `None` or not configured, only the duration on the Host side is marked.
  - When set to a valid stream, the durations on both the Host side and the corresponding Device side are marked.

- **`domain`** (`str`): Optional. Specified domain name, indicating that an instantaneous event is marked in the specified domain. The default value is `'default'`, indicating the default domain. If not set, the default domain is used.

## Return Values

None.

## Examples

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

- Using a `with` statement

  ```python
  with torch_npu.npu.mstx.annotate('my_code_range', cur_stream):
      my_code()
  ```

  In the preceding example, the `with` statement instruments the APIs executed within its scope and collects their durations.

- Using it as a decorator

  ```python
  @torch_npu.npu.mstx.annotate()
  def my_code():
      print("my_code start")
      my_code()
      print("my_code end")
  ```
  
  In the preceding example, the name of the decorated function is used as the message for the instrumentation task by default.
