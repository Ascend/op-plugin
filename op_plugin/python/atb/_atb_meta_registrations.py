import torch
from torch.library import Library, impl
from torch_npu.utils._error_code import ErrCode, ops_error

'''
Registering Meta implementations for custom atb ops
'''

#meta register implementation
m = Library("atb", "IMPL", "Meta")


@impl(m, "npu_multi_head_latent_attention")
def npu_multi_head_latent_attention_meta(q_nope, q_rope, ctkv, k_rope, block_tables, context_lens, q_headnum, qk_scale, kv_headnum,
                                            *, mask=None, qseqlen=None, qk_descale=None, pv_descale=None, mask_type=None, calc_type=None,
                                            cache_mode=None):
    return torch.empty_like(q_nope, dtype=q_rope.dtype)


@impl(m, "npu_self_attention_prefix_encoder")
def npu_self_attention_prefix_encoder_meta(query, key, value, block_tables, seqlen, kv_seqlen, q_headnum, qk_scale, kv_headnum, *,
                                            mask=None, slopes=None, mask_type=None):
    return torch.empty_like(query)


@impl(m, "npu_mla_preprocess")
def npu_mla_preprocess_meta(input, gamma0, beta0, wdqkv, descale0, gamma1, beta1, wuq, descale1, gamma2, cos, sin, wuk, kv_cache, kv_cache_rope,
                            slotmapping, *, quant_scale0=None, quant_offset0=None, bias0=None, quant_scale1=None, quant_offset1=None, bias1=None,
                            ctkv_scale=None, q_nope_scale=None, cache_mode=None, quant_mode=None):
    token_num = input.size(0)
    head_num = wuk.size(0)
    q_out0 = torch.empty([token_num, head_num, 512], dtype=kv_cache.dtype, device=input.device)
    q_out1 = torch.empty([token_num, head_num, 64], dtype=input.dtype, device=input.device)
    kv_cache_out0 = torch.empty_like(kv_cache)
    kv_cache_out1 = torch.empty_like(kv_cache_rope)
    return q_out0, kv_cache_out0, q_out1, kv_cache_out1


@impl(m, "npu_fused_add_topk_div")
def npu_fused_add_topk_div_meta(x, add_num, *, mapping_num=None, mapping_table=None, activation_type=None, group_num=1, group_topk=1, n=1, k=1, is_norm=True, scale=1, enable_expert_mapping=False):
    a = x.size(0)
    y = torch.empty([a, k], dtype=torch.float32, device=x.device)
    indices = torch.empty([a, k], dtype=torch.int32, device=x.device)
    return y, indices
