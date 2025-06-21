import torch_npu


def _add_torch_npu_atb_docstr(method, docstr):
    """Add doc to atb operator API.
    """
    func = getattr(torch_npu.atb, method, None)
    if not func:
        return
    getattr(torch_npu.atb, method).__doc__ = docstr


def _add_torch_npu_atb_api_docstr():
    _add_torch_npu_atb_docstr(
        "npu_multi_head_latent_attention",
        """
torch_npu.atb.npu_multi_head_latent_attention(q_nope, q_rope, ctkv, k_rope, block_tables, context_lens, q_headnum, qk_scale, kv_headnum, *, mask=None, qseqlen=None, qk_descale=None, pv_descale=None, mask_type=None, calc_type=None, cache_mode=None, output=None) -> Tensor
功能描述
MLA场景，使用分页管理的kvcache计算attention score，额外支持分离qnope/qrope、ctkv/krope的输入。

参数说明
位置参数：
q_nope: Device Tensor, 无位置编码query, 支持float16/bf16/int8， shape为[num_tokens, num_heads, 512]
q_rope: Device Tensor, 旋转位置编码query, 支持float16/bf16， shape为[num_tokens, num_heads, 64]
ctkv: Device Tensor, 无位置编码ctkv, 支持float16/bf16/int8数据类型，cache_mode为'krope_ctkv'时，shape为[num_blocks , block_size, kv_heads, 512]，cache_mode为'int8_nzcache'时，shape为[blockNum, kv_heads*512/32,block_size, 32]，数据类型为int8, 格式NZ。
k_rope: Device Tensor, 旋转位置编码k, 支持float16/bf16，cache_mode为'krope_ctkv', shape为[num_blocks , block_size, kv_heads, 64]，cache_mode为'int8_nzcache'时， shape为[blockNum, kv_heads*64 / 16 ,block_size, 16]，格式为NZ。
block_tables: Device Tensor, 每个query的kvcache的block映射表，shape为[batch, max_num_blocks_per_query]，数据类型为int32。
context_lens：int类型数组，每个query对应的上下文长度，kseqlen。
q_headnum：int类型，query头数量。
qk_scale: float类型，Q*K^T后乘以的缩放系数
kv_headnum：int类型，kv头数量。
关键字参数：
mask: 可选Device Tensor,  mask_type为默认场景时，可不传。
qseqlen: 可选Device Tensor, cache_mode为默认场景时不传，cache_mode为'int8_nzcache'时需要传入，shape为[num_heads]，数据类型为float。  
qk_descale: 可选Device Tensor,  cache_mode为默认场景时不传，cache_mode为'int8_nzcache'时需要传入，shape为[num_heads]，数据类型为float。
qk_descale: 可选Device Tensor,  cache_mode为默认场景时不传，cache_mode为'int8_nzcache'时需要传入，shape为[num_heads]，数据类型为float。
mask_type：可选字符串，设置mask类型，缺省值为'undefined', 当前支持'undefined'，无mask.
calc_type：可选字符串，设置mask类型，缺省值为'calc_type_undefined', 当前支持'calc_type_undefined'（默认的decoder场景。）.
cache_mode: 可选字符串，输入query和kcache的类型, 缺省值为'krope_ctkv'(输入的q拆分为qNope和qRope，输入的kcache拆分为ctKV和kRope), 当前支持'krope_ctkv', 'int8_nzcache'(性能cache，在KROPE_CTKV的基础上：krope和ctkv转为NZ格式输出，ctkv和qnope经过per_head静态对称量化为int8类型。)。
output: 可选Device Tensor，attention输出，shape为[num_tokens, num_heads, head_size_vo]，数据类型为float16/bf16。
规格约束
block_size ＜= 128，建议为128。
batch <= 8192
cache_mode为int8_nzcache, 不支持num_heads = 128

参考样例：
import torch,torch_npu
block_size = 128
num_tokens = 32
num_heads = 32
kv_heads =  1
head_size_qk = 576
head_size_vo = 512
batch = num_tokens
num_blocks = 64
max_num_blocks_per_query = 16

q_node = torch.randn((num_tokens, num_heads, head_size_vo), dtype=torch.float16).npu()
q_rope = torch.randn((num_tokens, num_heads, head_size_qk - head_size_vo), dtype=torch.float16).npu()
ctkv = torch.randn((num_blocks, block_size, kv_heads, 512), dtype=torch.float16).npu()
k_rope = torch.randn((num_blocks , block_size, kv_heads, 64), dtype=torch.float16).npu()
block_tables = torch.randint(0, 10, (batch, max_num_blocks_per_query), dtype=torch.int32).npu()
contextLens = [10] * batch

attenOut = torch_npu.atb.npu_multi_head_latent_attention(q_node, q_rope, ctkv, k_rope, block_tables, contextLens,32,1.0, 1)
    """
    )

    _add_torch_npu_atb_docstr(
        "npu_self_attention_prefix_encoder",
        """
torch_npu.atb.npu_self_attention_prefix_encoder(query, key, value, block_tables, seqlen, kv_seqlen, q_headnum, qk_scale, kv_headnum, *, mask=None, slopes=None, mask_type=None, output=None) -> Tensor
功能描述
MLA场景，使用分页管理的kvcache计算attention score，额外支持分离qnope/qrope、ctkv/krope的输入。

参数说明
位置参数：
query: Device Tensor, query矩阵, 支持float16/bf16， shape为[batch*qSeqLen, qHiddenSize]和[batch*qSeqLen, headNum, headSize]
key: Device Tensor, key矩阵, 支持float16/bf16， shape为[numBlocks, blockSize, kvHiddenSize]和[numBlocks, blockSize, headNum, headSize]
value: Device Tensor, 无位置编码ctkv, 支持float16/bf16数据类型，shape为[numBlocks, blockSize, kvHiddenSize]和[numBlocks, blockSize, headNum, headSize]	
block_tables: Device Tensor, 必选，每个query的kvcache的block table，第一维是token索引，第二维表示block索引
seqlen：int类型数组，query对应的每个batch的序列长度
kv_seqlen：int类型，key, value对应的每个batch的序列长度
q_headnum: int类型，query头数量。
qk_scale: float类型，Q*K^T后乘以的缩放系数
kv_headnum：int类型，kv头数量。
关键字参数：
mask: 可选Device Tensor,  默认场景下不需要输入
slopes: 可选Device Tensor,  默认场景下不需要输入
mask_type：可选字符串，设置mask类型，缺省值为'mask_type_causal_mask', 当前支持'mask_type_causal_mask'.
output: 可选Device Tensor，attention输出，shape为[batch*qSeqLen, qHiddenSize]或[batch*seqLen, headNum, headSize]，数据类型为float16/bf16。

约束：
qseqlen <= kv_seqlen
(kv_seqlen - q_seqlen) % 128 == 0

参考样例：
import torch,torch_npu
batch = 4
qseqlen = 128
headnum = 28
headsize = 128
numblocks = 64
block_size = 128
q_seqlens = [32] * batch
kv_seqLen = [32] * batch
dtype = torch.float16

query = torch.randn((batch*qseqlen, headnum, headsize), dtype=dtype).npu()
key = torch.randn((numblocks, block_size, headnum, headsize), dtype=dtype).npu()
value = torch.randn((numblocks, block_size, headnum, headsize), dtype=dtype).npu()
block_tables = torch.randint(0, 10, (batch, 64), dtype=torch.int32).npu()

output = torch_npu.atb.npu_self_attention_prefix_encoder(query, key, value, block_tables, q_seqlens, kv_seqLen, 28, 0.0883,4)
    """
    )
    _add_torch_npu_atb_docstr(
        "npu_mla_preprocess",
        """
torch_npu.atb.npu_mla_preprocess(input, gamma0, beta0, wdqkv, descale0, gamma1, beta1, wuq, descale1, gamma2, cos, sin, wuk, kv_cache, kv_cache_rope,
                                slotmapping, *,quant_scale0=None, quant_offset0=None, bias0=None, quant_scale1=None, quant_offset1=None, bias1=None,
                                ctkv_scale=None, q_nope_scale=None, cache_mode=None, quant_mode=None) -> (Tensor q_out0, Tensor kv_cache_out0, Tensor q_out1, Tensor kv_cache_out1)
功能描述
融合了MLA场景下PagedAttention输入数据处理的全过程，包括从隐状态输入开始经过rmsnorm、反量化、matmul、rope、reshapeAndCache的一系列计算。
默认场景：cache_mode为krope_ctkv，quant_mode为per_token_quant_symm

参数说明：下属参数说明基于融合的各个小算子输入进行说明。
位置参数：
input: Device Tensor, rmsNormQuant_0算子输入, 支持float16/bf16，shape为[tokenNum, 7168]
gamma0: Device Tensor, rmsNormQuant_0算子输入, 支持float16/bf16，shape为[7168], 数据类型与input一致
beta0: Device Tensor, rmsNormQuant_0算子输入, 支持float16/bf16，shape为[7168], 数据类型与input一致
wdqkv: Device Tensor, matmul_0算子输入, 支持int8，shape为[2112,7168]
descale0: Device Tensor, matmul_0算子输入, 支持int8，shape为[2112]
gamma1: Device Tensor, rmsNormQuant_1算子输入, 支持float16/bf16，shape为[1536], 数据类型与input一致
beta1: Device Tensor, rmsNormQuant_1算子输入, 支持float16/bf16，shape为[1536], 数据类型与input一致
wuq: Device Tensor, matmul_1算子输入, 支持int8，shape为[headNum * 192, 1536]
descale1: Device Tensor, matmul_1算子输入, input为float16时为int64，input为bf16时为float，shape为[headNum * 192]
gamma2: Device Tensor, rmsNorm算子输入, 支持float16/bf16，shape为[512], 数据类型与input一致
cos: Device Tensor, rope算子输入, 支持float16/bf16，shape为[tokenNum,64], 数据类型与input一致
sin: Device Tensor, rope算子输入, 支持float16/bf16，shape为[tokenNum,64], 数据类型与input一致
wuk: Device Tensor, matmulEin算子输入, 支持float16/bf16, 数据类型与input一致, ND格式时shape为[headNum,128,512], NZ格式时shape为[headNum,32,128,16]
kv_cache:  Device Tensor, reshapeAndCache算子输入, 支持float16/bf16/int8, cache_mode为'krope_ctkv'时，数据类型与input一致, shape为[blockNum,blockSize,1,512]， cache_mode为'int8_nzcache'时，需为NZ格式， int8数据类型，shape为[blockNum, headNum*512/32,block_size, 32]
kv_cache_rope: Device Tensor, reshapeAndCache算子输入, 支持float16/bf16 ，数据类型与input一致cache_mode为'krope_ctkv'时, shape为[blockNum,blockSize,1,64]， cache_mode为'int8_nzcache'时，需为NZ格式，shape为[blockNum, headNum*64 / 16 ,block_size, 16]
slotmapping: Device Tensor, reshapeAndCache算子输入, 支持int32, shape为[tokenNum]
关键字参数：
quant_scale0: 可选Device Tensor, rmsNormQuant_0算子输入，quant_mode为'per_token_quant_symm'时不需要输入, 支持float16/bf16，shape为[1], 数据类型与input一致
quant_offset0: 可选Device Tensor, rmsNormQuant_0算子输入，quant_mode为'per_token_quant_symm'时不需要输入, 支持int8，shape为[1]
bias0: 可选Device Tensor, matmul_0算子输入, 支持int8，shape为[2112]
quant_scale1: 可选Device Tensor, rmsNormQuant_1算子输入，quant_mode为'per_token_quant_symm'时不需要输入, 支持float16/bf16，shape为[1], 数据类型与input一致
quant_offset1: 可选Device Tensor, rmsNormQuant_1算子输入，quant_mode为'per_token_quant_symm'时不需要输入, 支持int8，shape为[1]
bias1: 可选Device Tensor, matmul_1算子输入，quant_mode为'per_token_quant_symm'时不需要输入, 支持int32，shape为[headNum * 192]
ctkv_scale: 可选Device Tensor,  默认场景下不需要输入, quant算子输入, shape为[1], 数据类型与input一致支持float16/bf16, cache_mode为'int8_nzcache'时,传入此tensor。
q_nope_scale: 可选Device Tensor,  默认场景下不需要输入, quant算子输入, shape为[headNum], 数据类型与input一致支持float16/bf16, cache_mode为'int8_nzcache'时,传入此tensor。
cache_mode: 可选字符串，输入query和kcache的类型, 缺省值为'krope_ctkv', 当前支持'krope_ctkv'（输入输出的kvCcache拆分为krope和ctkv，q拆分为qrope和qnope）, 'int8_nzcache'（krope和ctkv转为NZ格式输出，ctkv和qnope经过per_head静态对称量化为int8类型）。
quant_mode: 可选字符串，表示RmsNorm量化类型, 缺省值为'per_token_quant_symm'（per_token动态对称量化）, 当前支持'per_token_quant_symm', 'per_tensor_quant_asymm'（per_tensor静态非对称量化）。

约束：
tokenNum＜=1024。
blockSize <= 128 或 blockSize = 256。
cache_mode为int8_nzcache时，blockSize = 128。

参考样例（cache_mode为int8_nzcache，quant_mode为per_tensor_quant_asymm场景）：
import torch,torch_npu
token_num = 8
head_num = 128
N_7168 = 7168
block_num = 192
block_size = 128
dtype = torch.float16
device = 'npu'

input1 = torch.randn((token_num, N_7168), dtype=dtype, device=device)

gamma0 = torch.randn((N_7168), dtype=dtype, device=device)
beta0 = torch.randn((N_7168), dtype=dtype, device=device)
quant_scale0 = torch.randn((1,), dtype=dtype, device=device)
quant_offset0 = torch.randint(0, 7, (1,), dtype=torch.int8, device=device)

wdqkv = torch.randint(0, 7, (1, 224, 2112, 32), dtype=torch.int8, device=device)
wdqkv = torch_npu.npu_format_cast(wdqkv, 29)
de_scale0 = torch.randint(0, 7, (2112, ), dtype=torch.int64, device=device)
bias0 = torch.randint(0, 7, (2112, ), dtype=torch.int32, device=device)

gamma1 = torch.randn((1536), dtype=dtype, device=device)
beta1 = torch.randn((1536), dtype=dtype, device=device)
quant_scale1 = torch.randn((1,), dtype=dtype, device=device)
quant_offset1 = torch.randint(0, 7, (1,), dtype=torch.int8, device=device)

wuq = torch.randint(0, 7, (1, 48, head_num * 192, 32), dtype=torch.int8, device=device)
wuq = torch_npu.npu_format_cast(wuq, 29)
de_scale1 = torch.randint(0, 7, (head_num * 192, ), dtype=torch.int64, device=device)
bias1 = torch.randint(0, 7, (head_num * 192, ), dtype=torch.int32, device=device)

gamma2 = torch.randn((512), dtype=dtype, device=device)

cos = torch.randn((token_num, 64), dtype=dtype, device=device)
sin = torch.randn((token_num, 64), dtype=dtype, device=device)

wuk = torch.randn((head_num, 128, 512), dtype=dtype, device=device)
wuk = torch_npu.npu_format_cast(wuk, 29)

kv_cache = torch.randint(0, 7, (block_num, head_num * 512 // 32, block_size, 32), dtype=torch.int8, device=device)
kv_cache = torch_npu.npu_format_cast(kv_cache, 29)
kv_cache_rope = torch.randn((block_num, head_num * 64 // 16, block_size, 16), dtype=dtype, device=device)
kv_cache_rope = torch_npu.npu_format_cast(kv_cache_rope, 29)

slotmapping = torch.randint(0, 7, (token_num,), dtype=torch.int32, device=device)

ctkv_scale = torch.randn((1,), dtype=dtype, device=device)
qnope_scale = torch.randn((head_num), dtype=dtype, device=device)

q_out0 = torch.randint(0, 7, (token_num, head_num, 512), dtype=torch.int8, device=device)
kv_cache_out0 = torch.randint(0, 7, (block_num, head_num * 512 // 32, block_size, 32), dtype=torch.int8, device=device)
kv_cache_out0 = torch_npu.npu_format_cast(kv_cache_out0, 29)
q_out1 = torch.randn((token_num, head_num, 64), dtype=dtype, device=device)
kv_cache_out1 = torch.randn((block_num, head_num * 64 // 16, block_size, 16), dtype=dtype, device=device)
kv_cache_out1 = torch_npu.npu_format_cast(kv_cache_out1, 29)

torch_npu.atb.npu_mla_preprocess(
    input1, gamma0, beta0, wdqkv, de_scale0,
    gamma1, beta1, wuq, de_scale1,
    gamma2, cos, sin, wuk, kv_cache, kv_cache_rope, slotmapping,
    quant_scale0=quant_scale0,
    quant_offset0=quant_offset0,
    bias0=bias0,
    quant_scale1=quant_scale0,
    quant_offset1=quant_offset1,
    bias1=bias1,
    ctkv_scale=ctkv_scale,
    q_nope_scale=qnope_scale,
    cache_mode="int8_nzcache",
    quant_mode="per_tensor_quant_asymm",
    q_out0=q_out0,
    kv_cache_out0=kv_cache_out0,
    q_out1=q_out1,
    kv_cache_out1=kv_cache_out1,
)
    """
    )
