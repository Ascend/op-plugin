# torch_npu.npu_mla_prolog

> [!NOTICE]  
> This API is scheduled for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>   | √  |

## Function

- Description: Performs computation during Multi-Head Latent Attention (MLA) preprocessing in inference scenarios.

  The main computation flow consists of four separate paths:

  - First, input `x` is multiplied by W<sup>DQ</sup> for downsampling and processed through `RmsNorm`. This output is then split into two distinct paths. In the first path, the tensor is multiplied by W<sup>UQ</sup> and W<sup>UK</sup>, followed by two upsampling operations, to yield q<sup>N</sup>. In the second path, the tensor is multiplied by W<sup>QR</sup> and processed through rotary positional encoding (`ROPE`) to yield q<sup>R</sup>.
  - In the third path, input `x` is multiplied by W<sup>DKV</sup> for downsampling, processed through `RmsNorm`, and committed to a cache layer to yield k<sup>C</sup>.
  - In the fourth path, input `x` is multiplied by W<sup>KR</sup>, processed through `ROPE`, and committed to a separate cache layer to yield k<sup>R</sup>.
  
- Formulas:
    - `RmsNorm` formula:
        $$RmsNorm(x) = \gamma \cdot \frac{x} {\sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}}$$

    - Query computation formulas:
        $$c^Q = RmsNorm(x \cdot W^{DQ})$$
        $$q^C = c^Q \cdot W^{UQ}$$
        $$q^N = q^C \cdot W^{UK}$$

    - Query `ROPE` rotary positional encoding formula:
        $$q^R = ROPE(c^Q \cdot W^{QR})$$

    - Key computation formula:
        $$k^C = Cache(RmsNorm(x \cdot W^{DKV}))$$

    - Key `ROPE` rotary positional encoding formula:
        $$k^R = Cache(ROPE(x \cdot W^{KR}))$$

## Prototype

```python
torch_npu.npu_mla_prolog(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, *, dequant_scale_x=None, dequant_scale_w_dq=None, dequant_scale_w_uq_qr=None, dequant_scale_w_dkv_kr=None, quant_scale_ckv=None, quant_scale_ckr=None, smooth_scales_cq=None, rmsnorm_epsilon_cq=1e-05, rmsnorm_epsilon_ckv=1e-05, cache_mode="PA_BSND") -> (Tensor, Tensor, Tensor, Tensor)
```

## Parameters

- **`token_x`** (`Tensor`): Required. `x` in the formulas. This parameter must be 2D or 3D, with shape `(T, He)` or `(B, S, He)`. The data type can be `bfloat16`. The data layout can be ND.
- **`weight_dq`** (`Tensor`): Required. Downsampling weight matrix for query computation, W<sup>DQ</sup> in the formulas. This parameter must be 2D with shape `(He, Hcq)`. The data type can be `bfloat16`. The data layout can be `FRACTAL_NZ`. The layout can be converted from ND by using `torch_npu.npu_format_cast`.
- `weight_uq_qr` (`Tensor`): Required. Combined upsampling weight matrix and positional encoding weight matrix for query computation, W<sup>UQ</sup> and W<sup>QR</sup> in the formulas. This parameter must be 2D with shape `(Hcq, N * (D + Dr))`. The data type can be `bfloat16` or `int8`. The data layout can be FRACTAL\_NZ.  
    - When `weight_uq_qr` is set to `int8`, a per-tensor quantized input is specified. This configuration indicates a partial quantization scenario. 
        - If `kv_cache` and `kr_cache` are set to `bfloat16`, the outputs `kv_cache_out` and `kr_cache_out` are non-quantized outputs. In this case, `dequant_scale_w_uq_qr` must be provided, and `smooth_scales_cq` is optional. 
        - If `kv_cache` and `kr_cache` are set to `int8`, the outputs `kv_cache_out` and `kr_cache_out` are quantized outputs. In this case, `dequant_scale_w_uq_qr`, `quant_scale_ckv`, and `quant_scale_ckr` must be provided, and `smooth_scales_cq` is optional.   
    - When `weight_uq_qr` is set to `bfloat16`, a non-quantized scenario is specified.  
        In this configuration, `dequant_scale_w_uq_qr`, `quant_scale_ckv`, and `quant_scale_ckr`, and `smooth_scales_cq` cannot be provided and must be `None`.
  
- **`weight_uk`** (`Tensor`): Required. Upsampling weight matrix for key computation, corresponding to W<sup>UK</sup> in the formulas. This parameter must be 3D with shape `(N, D, Hckv)`. The data type can be `bfloat16`. The data layout can be ND.
- **`weight_dkv_kr`** (`Tensor`): Required. Combined downsampling weight matrix and positional encoding weight matrix for key computation, W<sup>DKV</sup> and W<sup>KR</sup> in the formulas. This parameter must be 2D with shape `(He, Hckv + Dr)`. The data type can be `bfloat16`. The data layout can be FRACTAL\_NZ.
- **`rmsnorm_gamma_cq`** (`Tensor`): Required. The $\gamma$ parameter in the `RmsNorm` formula for computing c<sup>Q</sup>. This parameter must be 1D with shape `(Hcq,)`. The data type can be `bfloat16`. The data layout can be ND.
- **`rmsnorm_gamma_ckv`** (`Tensor`): Required. The $\gamma$ parameter in the `RmsNorm` formula for computing c<sup>KV</sup>. This parameter must be 1D with shape `(Hckv,)`. The data type can be `bfloat16`. The data layout can be ND.
- **`rope_sin`** (`Tensor`): Required. Sine parameter matrix for rotary positional encoding (`ROPE`) computation. This parameter must be 2D or 3D with shape `(T, Dr)` or `(B, S, Dr)`. The data type can be `bfloat16`. The data layout can be ND.
- **`rope_cos`** (`Tensor`): Required. Cosine parameter matrix for rotary positional encoding (`ROPE`) computation. This parameter must be 2D or 3D with shape `(T, Dr)` or `(B, S, Dr)`. The data type can be `bfloat16`. The data layout can be ND.
- **`cache_index`** (`Tensor`): Required. Index for storing `kv_cache` and `kr_cache`. This parameter must be 1D or 2D with shape `(T,)` or `(B, S)`. The data type can be `int64`. The data layout can be ND. The value range of `cache_index` is `[0, BlockNum * BlockSize)`. The validity of input values is not verified internally, and must be ensured by the user.
- **`kv_cache`** (`Tensor`): Required. Cache tensor for key and value indexing. This parameter must be 4D with shape `(BlockNum, BlockSize, Nkv, Hckv)`. The data type can be `bfloat16` or `int8`. The data layout can be ND.
- **`kr_cache`** (`Tensor`): Required. Cache tensor for key rotary positional encoding. This parameter must be 4D with shape `(BlockNum, BlockSize, Nkv, Dr)`. The data type can be `bfloat16` or `int8`. The data layout can be ND.
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`dequant_scale_x`** (`Tensor`): Optional. Reserved parameter, currently not used. Retain the default value.
- **`dequant_scale_w_dq`** (`Tensor`): Optional. Reserved parameter, currently not used. Retain the default value.
- **`dequant_scale_w_uq_qr`** (`Tensor`): Optional. Dequantization scale parameter used after the `MatmulQcQr` matrix multiplication operation. The quantization mode is `perchannel`. This parameter must be 2D with shape `(1, N * (D + Dr))`. The data type can be `float`. The data layout can be ND.
- **`dequant_scale_w_dkv_kr`** (`Tensor`): Optional. Reserved parameter, currently not used. Retain the default value.
- **`quant_scale_ckv`** (`Tensor`): Optional. Quantization scale parameter used when writing data to `kv_cache_out`. This parameter must be 2D with shape `(1, Hckv)`. The data type can be `float`. The data layout can be ND.
- **`quant_scale_ckr`** (`Tensor`): Optional. Quantization scale parameter used when writing data to `kr_cache_out`. This parameter must be 2D with shape `(1, Dr)`. The data type can be `float`. The data layout can be ND.
- **`smooth_scales_cq`** (`Tensor`): Optional. Scaling factors used for dynamic quantization of the `RmsNormCq` output. This parameter must be 2D with shape `(1, Hcq)`. The data type can be `float`. The data layout can be ND.
- **`rmsnorm_epsilon_cq`** (`float`): Optional. The $\epsilon$ parameter in the `RmsNorm` formula for computing c<sup>Q</sup>. The default value is `1e-05`.
- **`rmsnorm_epsilon_ckv`** (`float`): Optional. The $\epsilon$ parameter in the `RmsNorm` formula for computing c<sup>KV</sup>. The default value is `1e-05`.
- **`cache_mode`** (`str`): Optional. Layout mode of `kv_cache`. The supported values are `"PA_BSND"` and `"PA_NZ"`. The default value is `"PA_BSND"`.

## Return Values

- **`query`** (`Tensor`): Query output tensor, corresponding to q<sup>N</sup> in the formulas. This parameter must be 3D or 4D, with shape `(T, N, Hckv)` or `(B, S, N, Hckv)`. The data type can be `bfloat16`. The data layout can be ND.
- **`query_rope`** (`Tensor`): Output tensor for query positional encoding, q<sup>R</sup> in the formulas. This parameter must be 3D or 4D with shape `(T, N, Dr)` or `(B, S, N, Dr)`. The data type can be `bfloat16`. The data layout can be ND.
- **`kv_cache_out`** (`Tensor`): Tensor written to `kv_cache` through an in-place update, k<sup>C</sup> in the formulas. This parameter must be 4D with shape `(BlockNum, BlockSize, Nkv, Hckv)`. The data type can be `bfloat16` or `int8`. The data layout can be ND.
- **`kr_cache_out`** (`Tensor`): Tensor written to `kr_cache` through an in-place update, corresponding to k<sup>R</sup> in the formulas. This parameter must be 4D with shape `(BlockNum, BlockSize, Nkv, Dr)`. The data type can be `bfloat16` or `int8`. The data layout can be ND.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.
- Field definitions for shape formats:
    - `B`: batch size, representing the input sample batch size. The value ranges from 0 to 65536.
    - `S`: seq-Length, representing the input sample sequence length. The value ranges from 0 to 16.
    - `He`: Head-Size, representing the hidden layer size. The value must be `7168`.

    - `Hcq`: dimension of the low-rank query matrix. The value must be `1536`.
    - `N`: head-Num, representing the attention head count. Supported values are `1`, `2`, `4`, `8`, `16`, `32`, `64`, and `128`.

    - `Hckv`: dimension of the low-rank key and value matrix. The value must be `512`.
    - `D`: dimension of query and key without positional encoding. The value must be `128`.
    - `Dr`: dimension of query and key positional encoding. The value must be `64`.
    - `Nkv`: attention head count for key and value. The value must be `1`.
    - `BlockNum`: number of blocks in the PagedAttention scenario. The value is calculated by dividing the product of `B` and `Skv` by `BlockSize`, and then rounding up to the nearest integer. Here, `Skv` represents the key and value sequence length. This value can be `0`.
    - `BlockSize`: block size in the PagedAttention scenario. Supported values are `16` and `128`.
    - `T`: size after the fusion of the `B` and `S` axes. The value ranges from 0 to 1048576. Note: When `B` and `S` axes are fused, `token_x`, `rope_sin`, and `rope_cos` are 2D parameters, `cache_index` is a 1D parameter, and `query` and `query_rope` are 3D parameters.
- Shape constraints:
    - One or more of `B`, `S`, `T`, and `Skv` can be `0`. That is, inputs whose shapes depend on these values can be empty tensors. Other inputs must not be empty tensors.
        - If `B`, `S`, or `T` is set to `0`, `query` and `query_rope` output empty tensors, and `kv_cache`, `kr_cache`, `kv_cache_out`, and `kr_cache_out` are not updated.
        - If `Skv` is set to `0`, `query` and `query_rope` are computed normally, while `kv_cache`, `kr_cache`, `kv_cache_out`, and `kr_cache_out` are not updated and output empty tensors.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    import math
    # Generate random data and send it to the NPU.
    B = 8
    He = 7168
    Hcq = 1536
    Hckv = 512
    N = 32
    D = 128
    Dr = 64
    Skv = 1024
    S = 2
    Nkv = 1
    BlockSize = 128
    BlockNum = 64
    token_x = torch.rand(B, S, He, dtype=torch.bfloat16).npu()
    w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
    w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
    w_uq_qr = torch.rand(Hcq, N * (D + Dr), dtype=torch.bfloat16).npu()
    w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
    w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
    w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
    w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
    rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
    rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
    rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    cache_index = torch.rand(B, S).to(torch.int64).npu()
    kv_cache = torch.rand(BlockNum, BlockSize, Nkv, Hckv, dtype=torch.bfloat16).npu()
    kr_cache = torch.rand(BlockNum, BlockSize, Nkv, Dr, dtype=torch.bfloat16).npu()
    rmsnorm_epsilon_cq = 1.0e-5
    rmsnorm_epsilon_ckv = 1.0e-5
    cache_mode = "PA_BSND"
    
    # Call the MlaProlog operator.
    query_mla, query_rope_mla, kv_cache_out_mla, kr_cache_out_mla = torch_npu.npu_mla_prolog(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
    print(query_mla)
    # Expected output of the preceding code sample:
    tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ..
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.bfloat16)
    ```

- Graph mode call

    ```python
    # Configure graph capture
    import torch
    import torch_npu
    import math
    import torchair as tng
    
    from torchair.configs.compiler_config import CompilerConfig
    import torch._dynamo
    TORCHDYNAMO_VERBOSE=1
    TORCH_LOGS="+dynamo"
    
    # Configure logging and debug settings for graph capture
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl
    
    # Generate data
    B = 8
    He = 7168
    Hcq = 1536
    Hckv = 512
    N = 32
    D = 128
    Dr = 64
    Skv = 1024
    S = 2
    Nkv = 1
    BlockSize = 128
    BlockNum = 64
    token_x = torch.rand(B, S, He, dtype=torch.bfloat16).npu()
    w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
    w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
    w_uq_qr = torch.rand(Hcq, N * (D + Dr), dtype=torch.bfloat16).npu()
    w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
    w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
    w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
    w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
    rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
    rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
    rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    cache_index = torch.rand(B, S).to(torch.int64).npu()
    kv_cache = torch.rand(BlockNum, BlockSize, Nkv, Hckv, dtype=torch.bfloat16).npu()
    kr_cache = torch.rand(BlockNum, BlockSize, Nkv, Dr, dtype=torch.bfloat16).npu()
    rmsnorm_epsilon_cq = 1.0e-5
    rmsnorm_epsilon_ckv = 1.0e-5
    cache_mode = "PA_BSND"
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self):
            return torch_npu.npu_mla_prolog(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
    
    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
        query_mla, query_rope_mla, kv_cache_out_mla, kr_cache_out_mla = torch_npu.npu_mla_prolog(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
        print("single op output:", query_mla)
        print("graph output:", graph_output)
        
    if __name__ == "__main__":
        MetaInfershape()
    
    # Expected output of the preceding code sample:
    single op output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ...,
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.bfloat16)
    
    graph output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ...,
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.bfloat16) 
    ```
