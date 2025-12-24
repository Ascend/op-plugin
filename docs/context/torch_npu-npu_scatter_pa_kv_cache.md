# torch_npu.npu_scatter_pa_kv_cache

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

更新KvCache中指定位置的`key`和`value`。

输入输出支持以下场景：

- 场景一：
    ```python
    key:[batch, num_head, k_head_size]
    value:[batch, num_head, v_head_size]
    key_cache:[num_blocks, num_head * k_head_size /last_dim_k, block_size, last_dim_k]
    value_cache:[num_blocks, num_head * v_head_size // last_dim_k, block_size, last_dim_k]
    slot_mapping:[batch]
    ```    

- 场景二：
    ```python    
    key:[batch*seq_len, num_head, k_head_size]
    value:[batch*seq_len, num_head, v_head_size]
    keyCache:[num_blocks, block_size, num_head, k_head_size]
    valueCache:[num_blocks, block_size, num_head, v_head_size]
    slotMapping:[batch * seq_len]
    ```    

    其中`k_head_size`与`v_head_size`可以不同，也可以相同。

- 场景三：
    ```python    
    key:[batch, seq_len, num_head, k_head_size]
    value:[batch, seq_len, num_head, v_head_size]
    key_cache:[num_blocks, block_size, 1, k_head_size]
    value_cache:[num_blocks, block_size, 1, k_head_size]
    slot_mapping:[batch, num_head]
    compress_lens:[batch, num_head]
    seq_lens:[batch]
    compress_seq_offsets:[batch * num_head]
    ```    

上述场景根据构造的参数来区别，符合第一种入参构造走场景一，符合第二种构造走场景二，符合第三种构造走场景三。场景一、场景二没有`compress_lens`、`seq_lens`、`compress_seq_offsets`这三个可选参数。

## 函数原型

```
torch_npu.npu_scatter_pa_kv_cache(key, value, key_cache, value_cache, slot_mapping, *, compress_lens=None, compress_seq_offsets=None, seq_lens=None) -> ()
```

## 参数说明

- **key**（`Tensor`）：必选参数。表示待更新的key值，当前step多个token的`key`，支持3维或4维。数据类型支持`float16`、`float`、`bfloat16`、`int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`hifloat8`、`float8_e5m2`、`float8_e4m3fn`，数据格式支持$ND$。
- **value**（`Tensor`）：必选参数，表示待更新的value值，当前step多个token的`value`，支持3维或4维，数据类型和数据格式与`key`保持一致。
- **key_cache**（`Tensor`）：必选参数，表示需要更新的key cache，当前layer的key cache，只支持4维，数据类型和数据格式与`key`保持一致。
- **value_cache**（`Tensor`）：必选参数，表示需要更新的value cache，当前layer的value cache。只支持4维，数据类型和数据格式与`key`保持一致。
- **slot_mapping**（`Tensor`）：必选参数，表示每个token key或value在cache中的存储偏移，数据类型支持`int32`和`int64`，数据格式支持$ND$。
- **compress_lens**（`Tensor`）：可选参数，表示压缩量，数据类型与`slot_mapping`一致，数据格式支持$ND$，默认值为None。
- **compress_seq_offsets**（`Tensor`）：可选参数，表示每个batch每个head的压缩起点，数据类型与`slot_mapping`一致，数据格式支持$ND$，默认值为None。
- **seq_lens**（`Tensor`）：可选参数，表示每个batch的实际seqLens，数据类型与`slot_mapping`一致，数据格式支持$ND$，默认值为None。

## 返回值说明

无返回值，`key_cache`和`value_cache`会被原地更新。

## 约束说明

- 输入参数不支持非连续；
- `key`、`value`、`key_cache`、`value_cache`的数据类型必须一致；
- `slot_mapping`、`compress_lens`、`compress_seq_offset`、`seq_lens`的数据类型必须一致；
- `slot_mapping`的值范围[0, num_blocks * block_size-1]，且`slot_mapping`内的元素值保证不重复，重复时不保证正确性；
- 当`key`和`value`都是3维，则`key`和`value`的前两维`shape`必须相同；
- 当`key`和`value`都是4维，则`key`和`value`的前三维`shape`必须相同，且`key_cache`和`value_cache`的第三维必须是1；
- 当`key`和`value`是4维时，`compress_lens`、`seq_lens`为必选参数；当`key`和`value`是3维时，`compress_lens`、`compress_seq_offsets`、`seq_lens`为可选参数；
- 当`key`和`value`都是4维时，`slot_mapping`是二维，且`slot_mapping`的第一维值等于`key`的第一维为`batch`，`slot_mapping`的第二维值等于`key`的第三维为num_head(对应场景三)；
- 当`key`和`value`都是4维时，`seq_lens`是一维，且`seq_lens`的值等于`key`的第一维为batch(对应场景三)；
- `seq_lens`和`compress_lens`里面的每个元素值必须满足公式：reduceSum(seq_lens[i] - compress_lens[i]) <= num_blocks * block_size(对应场景三);

## 调用示例

- 单算子模式调用

    ```python
    >>> import numpy as np
    >>> import torch
    >>> import torch_npu
    >>>
    >>> bs = 16
    >>> num_head = 4
    >>> k_head_size = 32
    >>> v_head_size = 64
    >>> num_blocks = 2
    >>> lastDim_k = 16
    >>> block_size = 32
    >>>
    >>> key = np.random.randn(bs, num_head, k_head_size).astype(np.float16)
    >>> value = np.random.randn(bs, num_head, v_head_size).astype(np.float16)
    >>> key_cache = np.random.randn(
    >>>     num_blocks, num_head * k_head_size // lastDim_k, block_size, lastDim_k).astype(np.float16)
    >>> value_cache = np.zeros(
    >>>     (num_blocks, num_head * v_head_size // lastDim_k, block_size, lastDim_k)).astype(np.float16)
    >>> slot_mapping = np.random.choice(num_blocks * block_size, bs, replace=False).astype(np.int32)
    >>>
    >>> key_npu = torch.from_numpy(key).npu()
    >>> value_npu = torch.from_numpy(value).npu()
    >>> key_cache_npu = torch.from_numpy(key_cache).npu()
    >>> value_cache_npu = torch.from_numpy(value_cache).npu()
    >>> key_cache_npu_cast = torch_npu.npu_format_cast(key_cache_npu.contiguous(), 29)
    >>> value_cache_npu_cast = torch_npu.npu_format_cast(value_cache_npu.contiguous(), 29)
    >>> slot_mapping_npu = torch.from_numpy(slot_mapping).npu()
    >>>
    >>> torch_npu.npu_scatter_pa_kv_cache(key_npu, value_npu, key_cache_npu_cast, value_cache_npu_cast, slot_mapping_npu)
    >>> key_cache_npu_cast
    tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [[ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],
        ...,
        [[ 1.8271,  1.4551,  1.3154,  ...,  1.9854,  1.4365,  1.0732],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]],
        ...,
        [[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 1.9492,  1.6455,  1.6504,  ...,  1.5957,  1.6201,  1.4385],
        [ 0.0742,  0.1982,  0.8945,  ...,  0.4912,  0.6753,  0.1120],
        ...,
        [[ 0.1113,  0.6255,  0.7686,  ...,  0.0247,  0.2490,  0.6909],
        [ 0.4312,  0.7954,  0.7339,  ...,  0.1154,  0.6440,  0.3342],
        [ 0.9570,  0.2869,  0.6489,  ...,  0.7451,  0.0234,  0.8843]],
        ...,
        [[ 1.8271,  1.4551,  1.3154,  ...,  1.9854,  1.4365,  1.0732],
        [ 1.9492,  1.6455,  1.6504,  ...,  1.5957,  1.6201,  1.4385],
        [ 0.0742,  0.1982,  0.8945,  ...,  0.4912,  0.6753,  0.1120]]]]
        device='npu:0', dtype=torch.float16)
    ```

- 图模式调用

    ```python
    import numpy as np
    import torch
    import torch_npu

    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import torch._dynamo
    TORCHDYNAMO_VERBOSE=1
    TORCH_LOGS="+dynamo"

    # 支持入图的打印宏
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.aoe_config.aoe_mode = "2"
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl

    # 数据生成
    bs = 16
    num_head = 4
    k_head_size = 32
    v_head_size = 64
    num_blocks = 2
    lastDim_k = 16
    block_size = 32

    class Model(torch.nn.Module):
        def init(self):
            super().init()

        def forward(self, key, value, slot_mapping, key_cache, value_cache):
            torch_npu.npu_scatter_pa_kv_cache(key, value, key_cache, value_cache, slot_mapping)

    if name=="main":
        torch_npu.npu.set_device(0)

        key = np.random.randn(bs, num_head, k_head_size).astype(np.float16)
        value = np.random.randn(bs, num_head, v_head_size).astype(np.float16)
        key_cache = np.random.randn(
            num_blocks, num_head * k_head_size // lastDim_k, block_size, lastDim_k).astype(np.float16)
        value_cache = np.zeros(
            (num_blocks, num_head * v_head_size // lastDim_k, block_size, lastDim_k)).astype(np.float16)
        slot_mapping = np.random.choice(num_blocks * block_size, bs, replace=False).astype(np.int32)

        key_npu = torch.from_numpy(key).npu()
        value_npu = torch.from_numpy(value).npu()
        key_cache_npu = torch.from_numpy(key_cache).npu()
        value_cache_npu = torch.from_numpy(value_cache).npu()
        key_cache_npu_cast = torch_npu.npu_format_cast(key_cache_npu.contiguous(), 29)
        value_cache_npu_cast = torch_npu.npu_format_cast(value_cache_npu.contiguous(), 29)
        slot_mapping_npu = torch.from_numpy(slot_mapping).npu()

        config = CompilerConfig()
        npu_backend = tng.get_npu_backend(compiler_config=config)

        cpu_model = Model().npu()
        # 图模式调用
        model = torch.compile(cpu_model, backend=npu_backend, dynamic=False, fullgraph=True)
        model(key_npu, value_npu, slot_mapping_npu, key_cache_npu_cast, value_cache_npu_cast)

        # 这里的输出与单算子一致
    ```
