import math
import torch
from torch.library import Library, impl
from torch.fx.node import has_side_effect
from torch_npu.utils._error_code import ErrCode, ops_error

'''
Registering Meta implementations for custom ops
'''
BIT_NUMBER = 128
UINT8_BIT_NUMBER = 8
INPUTS_DIM_LIMIT_QUANTCONV2D = 4
ATTR_DIM_LIMIT_QUANTCONV2D = 2
#meta register implementation
m = Library("npu", "IMPL", "Meta")
m_aten = Library("aten", "IMPL", "Meta")


@impl(m_aten, "matmul_backward")
def matmul_backward_meta(grad, self, other, mask):
    return (torch.empty_like(self), torch.empty_like(other))


@impl(m, "npu_incre_flash_attention")
def npu_incre_flash_attention_forward(query, key, value, *, padding_mask=None, atten_mask=None, pse_shift=None, actual_seq_lengths=None,
                                      antiquant_scale=None, antiquant_offset=None, block_table=None,
                                      dequant_scale1=None, quant_scale1=None, dequant_scale2=None, quant_scale2=None,
                                      quant_offset2=None, kv_padding_size=None, num_heads=1, scale_value=1.0, input_layout="BSH",
                                      num_key_value_heads=0, block_size=0, inner_precise=1):
    if quant_scale2 is not None:
        return torch.empty_like(query, dtype=torch.int8)
    elif query.dtype == torch.int8:
        return torch.empty_like(query, dtype=torch.half)
    else:
        return torch.empty_like(query)


@impl(m, "npu_mla_prolog")
def npu_mla_prolog_forward(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
                   rope_sin, rope_cos, cache_index, kv_cache, kr_cache, *, dequant_scale_x=None, dequant_scale_w_dq=None, dequant_scale_w_uq_qr=None, dequant_scale_w_dkv_kr=None,
                   quant_scale_ckv=None, quant_scale_ckr=None, smooth_scales_cq=None,
                   rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5, cache_mode="PA_BSND"):
    require_param = {"token_x": token_x, "weight_dq": weight_dq, "weight_uq_qr": weight_uq_qr, "weight_uk": weight_uk, "weight_dkv_kr": weight_dkv_kr, "rmsnorm_gamma_cq": rmsnorm_gamma_cq, "rmsnorm_gamma_ckv": rmsnorm_gamma_ckv, "rope_sin": rope_sin, "rope_cos": rope_cos, "cache_index": cache_index, "kv_cache": kv_cache, "kr_cache": kr_cache}

    for item_name, item in require_param.items():
        torch._check(
            item is not None,
            lambda: item_name + " should not be null,but the actual value is null" + ops_error(ErrCode.VALUE),
        )

    token_x_dim = token_x.dim()
    torch._check(
        token_x_dim == 3,
        lambda: "token_x dim num should be 3,but the actual value is " + str(token_x_dim) + ops_error(ErrCode.VALUE),
    )

    weight_uk_dim = weight_uk.dim()
    torch._check(
        weight_uk_dim == 3,
        lambda: "weight_uk dim num should be 3,but the actual value is " + str(weight_uk_dim) + ops_error(ErrCode.VALUE),
    )

    rope_sin_dim = rope_sin.dim()
    torch._check(
        rope_sin_dim == 3,
        lambda: "rope_sin dim num should be 3,but the actual value is " + str(rope_sin_dim) + ops_error(ErrCode.VALUE),
    )

    query_shape = []
    query_shape.append(token_x.size(0))
    query_shape.append(token_x.size(1))
    query_shape.append(weight_uk.size(0))
    query_shape.append(weight_uk.size(2))

    query_rope_shape = []
    query_rope_shape.append(token_x.size(0))
    query_rope_shape.append(token_x.size(1))
    query_rope_shape.append(weight_uk.size(0))
    query_rope_shape.append(rope_sin.size(2))

    query = torch.empty(query_shape, dtype=rope_sin.dtype, device='meta')
    query_rope = torch.empty(query_rope_shape, dtype=rope_sin.dtype, device='meta')
    kv_cache_out = torch.empty_like(kv_cache, dtype=kv_cache.dtype, device='meta')
    kr_cache_out = torch.empty_like(kr_cache, dtype=kr_cache.dtype, device='meta')

    return (query, query_rope, kv_cache_out, kr_cache_out)


if "2.1" in torch.__version__:
    @impl(m, "npu_prompt_flash_attention")
    def npu_prompt_flash_attention_forward(query, key, value, *, padding_mask=None, atten_mask=None, pse_shift=None, actual_seq_lengths=None, deq_scale1=None, quant_scale1=None, deq_scale2=None, quant_scale2=None, quant_offset2=None, num_heads=1, scale_value=1.0, pre_tokens=2147473647, next_tokens=0, input_layout="BSH", num_key_value_heads=0, actual_seq_lengths_kv=None, sparse_mode=0):
        tmp_out = torch.empty_like(query, dtype=query.dtype, device='meta')
        if input_layout == "BNSD_BSND":
            tmp_out = torch.empty([query.size(0), query.size(2), query.size(1), query.size(3)], dtype=query.dtype, device='meta')
        elif input_layout == "SH":
            tmp_out = torch.empty([query.size(0), query.size(1)], dtype=query.dtype, device='meta')
        elif input_layout == "BSH" or input_layout == "NSD":
            tmp_out = torch.empty([query.size(0), query.size(1), query.size(2)], dtype=query.dtype, device='meta')
        elif input_layout == "TND":
            tmp_out = torch.empty([query.size(0), query.size(1), value.size(2)], dtype=query.dtype, device='meta')
        else:
            tmp_out = torch.empty([query.size(0), query.size(1), query.size(2), query.size(3)], dtype=query.dtype, device='meta')
        if quant_scale2 is not None:
            return torch.empty_like(tmp_out, dtype=torch.int8)
        elif query.dtype == torch.int8:
            return torch.empty_like(tmp_out, dtype=torch.half)
        else:
            return torch.empty_like(tmp_out, dtype=query.dtype)
else:
    @impl(m, "npu_prompt_flash_attention")
    def npu_prompt_flash_attention_forward(query, key, value, *, padding_mask=None, atten_mask=None, pse_shift=None, actual_seq_lengths=None, deq_scale1=None, quant_scale1=None, deq_scale2=None, quant_scale2=None, quant_offset2=None, num_heads=1, scale_value=1.0, pre_tokens=2147473647, next_tokens=0, input_layout="BSH", num_key_value_heads=0, actual_seq_lengths_kv=None, sparse_mode=0):
        tmp_out = torch.empty_like(query, dtype=query.dtype, device='meta')
        if input_layout == "TND":
            tmp_out = torch.empty([query.size(0), query.size(1), value.size(2)], dtype=query.dtype, device='meta')

        if quant_scale2 is not None:
            return torch.empty_like(tmp_out, dtype=torch.int8)
        elif query.dtype == torch.int8:
            return torch.empty_like(tmp_out, dtype=torch.half)
        else:
            return torch.empty_like(tmp_out, dtype=query.dtype)


@impl(m, "npu_mm_reduce_scatter_base")
def npu_mm_reduce_scatter_base_meta(self, x2, hcom, world_size, reduce_op='sum',
                                    bias=None, comm_turn=0):
    if world_size <= 0:
        world_size = 1
    out_m = math.floor(self.size(0) / world_size)
    return self.new_empty(out_m, x2.size(1))


@impl(m, "npu_all_gather_base_mm")
def npu_all_gather_base_mm_meta(self, x2, hcom, world_size, bias=None,
                                gather_index=0, gather_output=True, comm_turn=0):
    if world_size <= 0:
        world_size = 1
    # out_gather_mm
    out_x = self.size(0)
    if gather_index == 0:
        out_x = self.size(0) * world_size
    out_y = x2.size(1)
    # out_gather
    out_gather_x = x2.size(0) * world_size
    out_gather_y = x2.size(1)
    if gather_index == 0:
        out_gather_x = self.size(0) * world_size
        out_gather_y = self.size(1)
    if gather_output:
        return (self.new_empty((out_x, out_y)), self.new_empty(out_gather_x, out_gather_y))
    else:
        return (self.new_empty((out_x, out_y)), self.new_empty(0))


@impl(m, "npu_moe_init_routing")
def npu_moe_init_routing_meta(x, row_idx, expert_idx, active_num=99):
    n = x.size(0)
    h = x.size(1)
    k = row_idx.size(1)
    active_num = min(n, active_num)
    expanded_x_dim_list = [active_num * k, h]
    expanded_row_idx_dim_list = [n * k]
    expanded_expert_idx_dim_list = [n * k]
    return (x.new_empty(tuple(expanded_x_dim_list)), row_idx.new_empty(tuple(expanded_row_idx_dim_list)), row_idx.new_empty(tuple(expanded_row_idx_dim_list)))


@impl(m, "npu_moe_init_routing_v2")
def npu_moe_init_routing_v2_meta(x, expert_idx, *, scale=None, offset=None, active_num=-1, expert_capacity=-1, expert_num=-1, drop_pad_mode=0, expert_tokens_num_type=0, expert_tokens_num_flag=False, quant_mode=0, active_expert_range=[], row_idx_type=0):
    x_dim = x.dim()
    torch._check(
        x_dim == 2,
        lambda: "the x shape support only 2d" + ops_error(ErrCode.VALUE),
    )
    expert_idx_dim = expert_idx.dim()
    torch._check(
        expert_idx_dim == 2,
        lambda: "the expert_idx shape support only 2d" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        x.size(0) == expert_idx.size(0),
        lambda: "the first dim of expert_idx and x should be the same" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        active_expert_range is not None and isinstance(active_expert_range, list) and len(active_expert_range) == 2,
        lambda: "active_expert_range is None or invalid. must be int[2]"
    )
    torch._check(
        active_expert_range[1] > active_expert_range[0],
        lambda: "active_expert_range is invalid. must be increasing"
    )
    torch._check(
        active_expert_range[0] >= 0 and active_expert_range[1] <= 10240,
        lambda: "active_expert_range must be within [0, 10240]"
    )
    expert_range_length = active_expert_range[1] - active_expert_range[0]

    torch._check(
        drop_pad_mode is not None and isinstance(drop_pad_mode, int) and drop_pad_mode in [0, 1],
        lambda: "drop_pad_mode is None or invalid. must be in [0, 1]"
    )
    torch._check(
        expert_tokens_num_type is not None and isinstance(expert_tokens_num_type, int) and expert_tokens_num_type in [0, 1],
        lambda: "expert_tokens_num_type is None or invalid. must be in [0, 1]"
    )
    torch._check(
        quant_mode is not None and isinstance(quant_mode, int) and quant_mode in [-1, 0, 1],
        lambda: "quant_mode is None or invalid. must be in [-1, 0, 1]"
    )
    torch._check(
        row_idx_type is not None and isinstance(row_idx_type, int) and row_idx_type in [0, 1],
        lambda: "row_idx_type is None or invalid. must be in [0, 1]"
    )

    if scale is not None:
        scale_dim = scale.dim()
        if quant_mode == -1:
            torch._check(
                scale_dim == 1,
                lambda: "the scale shape support only 1D (bs,) in no quant mode" + ops_error(ErrCode.VALUE),
            )
            torch._check(
                x.size(0) == scale.size(0),
                lambda: "the first dim of scale and the first dim of x should be the same" + ops_error(ErrCode.VALUE),
            )
        elif quant_mode == 0:
            torch._check(
                scale_dim == 2 or scale_dim == 1,
                lambda: "the scale shape should be (end-start, 1) or (end-start,) in static quant mode" + ops_error(ErrCode.VALUE),
            )
            torch._check(
                expert_range_length == scale.size(0),
                lambda: "the first dim of scale and expert_range_length should be the same" + ops_error(ErrCode.VALUE),
            )
            torch._check(
                scale_dim == 1 or x.size(1) == scale.size(1) or 1 == scale.size(1),
                lambda: "the 2nd dim of scale should be 1 or the same with the 2nd dim of x" + ops_error(ErrCode.VALUE),
            )
            if offset is not None:
                offset_dim = offset.dim()
                torch._check(
                    offset_dim == 2 or offset_dim == 1,
                    lambda: "the offset shape should be (end-start, 1) or (end-start,)" + ops_error(ErrCode.VALUE),
                )
                torch._check(
                    scale.size(0) == offset.size(0),
                    lambda: "the 1st dim of offset and the 1st dim of scale should be the same" + ops_error(ErrCode.VALUE),
                )
                torch._check(
                    offset_dim == 1 or x.size(1) == offset.size(1) or 1 == offset.size(1),
                    lambda: "the 2nd dim of offset and the 2nd dim of scale should be the same" + ops_error(ErrCode.VALUE),
                )
        else:
            torch._check(
                scale_dim == 2,
                lambda: "the scale shape support only 2D in dynamic quant mode" + ops_error(ErrCode.VALUE),
            )
            torch._check(
                expert_range_length == scale.size(0),
                lambda: "the first dim of scale and expert_range_length should be the same" + ops_error(ErrCode.VALUE),
            )
            torch._check(
                x.size(1) == scale.size(1),
                lambda: "the 2nd dim of scale should be the same with the 2nd dim of x" + ops_error(ErrCode.VALUE),
            )

    bs = x.size(0)
    h = x.size(1)
    k = expert_idx.size(1)
    expanded_x_dim_list = [bs * k, h]
    expanded_x_dtype = x.dtype if quant_mode == -1 else torch.int8
    expanded_row_idx_dim_list = [bs * k]
    expert_token_cumsum_or_count_dim_list = [expert_range_length]
    expanded_scale_dim_list = [bs * k]
    return (x.new_empty(tuple(expanded_x_dim_list), dtype=expanded_x_dtype),
            x.new_empty(tuple(expanded_row_idx_dim_list), dtype=torch.int32),
            x.new_empty(tuple(expert_token_cumsum_or_count_dim_list), dtype=torch.int64),
            x.new_empty(tuple(expanded_scale_dim_list), dtype=torch.float32))


@impl(m, "npu_moe_gating_top_k_softmax")
def npu_moe_gating_top_k_softmax_meta(x, finished=None, k=1):
    x_dim = x.dim()
    torch._check(
        x_dim == 2 or x_dim == 3,
        lambda: "the x shape support only 2d and 3d)" + ops_error(ErrCode.VALUE),
    )
    if x_dim == 3:
        y_dim_list = [x.size(0), x.size(1), k]
        expert_idx_dim_list = [x.size(0), x.size(1), k]
        row_idx_dim_list = [x.size(0), x.size(1), k]
    else:
        y_dim_list = [x.size(0), k]
        expert_idx_dim_list = [x.size(0), k]
        row_idx_dim_list = [x.size(0), k]
    return (x.new_empty(tuple(y_dim_list), dtype=x.dtype),
            x.new_empty(tuple(expert_idx_dim_list), dtype=torch.int32),
            x.new_empty(tuple(row_idx_dim_list), dtype=torch.int32))


@impl(m, "npu_moe_gating_top_k")
def npu_moe_gating_top_k_meta(x, k=1, bias=None, k_group=1, group_count=1, group_select_mode=0, renorm=0, norm_type=0, out_flag=False, routed_scaling_factor=1.0, eps=1e-20):
    x_dim = x.dim()
    torch._check(
        x_dim == 2,
        lambda: "the x shape support only 2d)" + ops_error(ErrCode.VALUE),
    )
    if bias is not None:
        bias_dim = bias.dim()
        torch._check(
            bias_dim == 1,
            lambda: "the bias shape support only 1d)" + ops_error(ErrCode.VALUE),
        )
    y_dim_list = [x.size(0), k]
    expert_idx_dim_list = [x.size(0), k]
    y2_dim_list = [x.size(0), x.size(1)]
    return (x.new_empty(tuple(y_dim_list), dtype=x.dtype),
            x.new_empty(tuple(expert_idx_dim_list), dtype=torch.int32),
            x.new_empty(tuple(y2_dim_list), dtype=torch.float32))


@impl(m, "npu_fused_infer_attention_score")
def npu_fused_infer_attention_score_forward(query, key, value, *, pse_shift=None, atten_mask=None, actual_seq_lengths=None, actual_seq_lengths_kv=None,
                                    dequant_scale1=None, quant_scale1=None, dequant_scale2=None, quant_scale2=None,
                                    quant_offset2=None, antiquant_scale=None, antiquant_offset=None, block_table=None,
                                    query_padding_size=None, kv_padding_size=None, key_antiquant_scale=None, key_antiquant_offset=None,
                                    value_antiquant_scale=None, value_antiquant_offset=None, key_shared_prefix=None, value_shared_prefix=None,
                                    actual_shared_prefix_len=None, query_rope=None, key_rope=None, key_rope_antiquant_scale=None, num_heads=1, scale=1.0, pre_tokens=2147483647, next_tokens=2147483647,
                                    input_layout="BSH", num_key_value_heads=0, sparse_mode=0, inner_precise=0, block_size=0, antiquant_mode=0,
                                    softmax_lse_flag=False, key_antiquant_mode=0, value_antiquant_mode=0):
    tmp_out = torch.empty_like(query, dtype=query.dtype, device='meta')
    B = 1
    N = 1
    S1 = 1
    if input_layout == "BNSD_BSND":
        tmp_out = torch.empty([query.size(0), query.size(2), query.size(1), query.size(3)], dtype=query.dtype, device='meta')
        B = query.size(0)
        N = query.size(1)
        S1 = query.size(2)
    if input_layout == "BNSD":
        tmp_out = torch.empty([query.size(0), query.size(1), query.size(2), query.size(3)], dtype=query.dtype, device='meta')
        B = query.size(0)
        N = query.size(1)
        S1 = query.size(2)
    if input_layout == "BSH":
        tmp_out = torch.empty([query.size(0), query.size(1), query.size(2)], dtype=query.dtype, device='meta')
        B = query.size(0)
        N = num_heads
        S1 = query.size(1)
    if input_layout == "BSND":
        tmp_out = torch.empty([query.size(0), query.size(1), query.size(2), query.size(3)], dtype=query.dtype, device='meta')
        B = query.size(0)
        N = num_heads
        S1 = query.size(1)
    if input_layout == "NSD":
        tmp_out = torch.empty([query.size(0), query.size(1), query.size(2)], dtype=query.dtype, device='meta')
        B = 1
        N = query.size(0)
        S1 = query.size(1)
    if input_layout == "TND":
        tmp_out = torch.empty([query.size(0), query.size(1), value.size(2)], dtype=query.dtype, device='meta')

    if quant_scale2 is not None:
        if (softmax_lse_flag == True):
            if input_layout == "TND":
                return (torch.empty_like(tmp_out, dtype=torch.int8), torch.empty([query.size(0), num_heads, 1], dtype=torch.float32, device='meta'))
            else:
                return (torch.empty_like(tmp_out, dtype=torch.int8), torch.empty([B, N, S1, 1], dtype=torch.float32, device='meta'))
        else:
            return (torch.empty_like(tmp_out, dtype=torch.int8), torch.empty([1], dtype=torch.float32, device='meta'))
    elif query.dtype == torch.int8:
        if (softmax_lse_flag == True):
            if input_layout == "TND":
                return (torch.empty_like(tmp_out, dtype=torch.half), torch.empty([query.size(0), num_heads, 1], dtype=torch.float32, device='meta'))
            else:
                return (torch.empty_like(tmp_out, dtype=torch.half), torch.empty([B, N, S1, 1], dtype=torch.float32, device='meta'))
        else:
            return (torch.empty_like(tmp_out, dtype=torch.half), torch.empty([1], dtype=torch.float32, device='meta'))
    else:
        if (softmax_lse_flag == True):
            if input_layout == "TND":
                return (torch.empty_like(tmp_out), torch.empty([query.size(0), num_heads, 1], dtype=torch.float32, device='meta'))
            else:
                return (torch.empty_like(tmp_out), torch.empty([B, N, S1, 1], dtype=torch.float32, device='meta'))
        else:
            return (torch.empty_like(tmp_out), torch.empty([1], dtype=torch.float32, device='meta'))


@impl(m, "npu_fusion_attention")
def npu_fusion_attention_forward(query, key, value, head_num, input_layout, pse=None, padding_mask=None,
                                atten_mask=None, scale=1.0, keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647,
                                inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False):
    B = query.size(0)
    N = head_num
    S1 = query.size(2)
    S2 = key.size(2)

    if input_layout == "BSH":
        B = query.size(0)
        S1 = query.size(1)
        S2 = key.size(1)

    if input_layout == "SBH":
        B = query.size(1)
        S1 = query.size(0)
        S2 = key.size(0)

    seed = 0
    offset = 0
    numels = 0
    attention_score = torch.empty_like(query, dtype=query.dtype, device='meta')
    softmax_max = torch.empty([B, head_num, S1, 8], dtype=torch.float32, device='meta')
    softmax_sum = torch.empty([B, head_num, S1, 8], dtype=torch.float32, device='meta')
    softmax_out = torch.empty([0], dtype=query.dtype, device='meta')
    return (torch.empty_like(attention_score),
            torch.empty_like(softmax_max),
            torch.empty_like(softmax_sum),
            torch.empty_like(softmax_out),
            seed,
            offset,
            numels)


@impl(m, "npu_fusion_attention_grad")
def npu_fusion_attention_backward(query, key, value, dy, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None,
                                  softmax_max=None, softmax_sum=None, softmax_in=None, attention_in=None, scale_value=1.0,
                                  keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0, seed=0, offset=0,
                                  numels=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False):
    dq = torch.empty_like(query, dtype=query.dtype, device='meta')
    dk = torch.empty_like(key, dtype=query.dtype, device='meta')
    dv = torch.empty_like(value, dtype=query.dtype, device='meta')
    dpse = torch.empty([0], dtype=query.dtype, device='meta')
    return (torch.empty_like(dq), torch.empty_like(dk), torch.empty_like(dv), torch.empty_like(dpse))


@impl(m, "npu_rotary_mul")
def npu_rotary_mul_meta(embedding, cosine, sine, mode='half'):
    return torch.empty_like(embedding)


@impl(m, "npu_rotary_mul_backward")
def npu_rotary_mul_backward(grad, embedding, cosine, sine, mode=0):
    dx = torch.empty_like(embedding, dtype=embedding.dtype, device='meta')
    dr1 = torch.empty_like(cosine, dtype=embedding.dtype, device='meta')
    dr2 = torch.empty_like(sine, dtype=embedding.dtype, device='meta')
    return (dx, dr1, dr2)


@impl(m, "fast_gelu")
def fast_gelu_meta(self):
    return torch.empty_like(self)


@impl(m, "npu_fast_gelu_backward")
def npu_fast_gelu_backward_meta(grad, self):
    return torch.empty_like(self)


@impl(m, "npu_fast_gelu")
def npu_fast_gelu_meta(self):
    return torch.empty_like(self)


@impl(m, "npu_gelu")
def npu_gelu_meta(self, *, approximate="none"):
    return torch.empty_like(self)


@impl(m, "npu_gelu_backward")
def npu_gelu_backward_meta(grad, self, *, approximate="none"):
    return torch.empty_like(self)


@impl(m, "npu_dtype_cast")
def npu_dtype_cast_meta(self, dtype):
    return torch.empty_like(self, dtype=dtype)


@impl(m, "npu_dtype_cast_backward")
def npu_dtype_cast_backward_meta(self, dtype):
    return torch.empty_like(self, dtype=dtype)


@impl(m, "npu_bmmV2")
def npu_bmmV2_meta(self, mat2, output_sizes):
    dim1 = self.size(0)
    dim2 = self.size(1)
    dim3 = mat2.size(2)
    return self.new_empty((dim1, dim2, dim3))


@impl(m, "npu_transpose")
def npu_transpose_meta(self, perm, require_contiguous=True):
    output = self.permute(perm)
    return torch.empty_like(output, dtype=self.dtype)


@impl(m, "npu_deep_norm")
def npu_deep_norm_meta(self, gx, beta, gamma, alpha=0.3, epsilon=1e-6):
    rstd_dim = self.dim() - gamma.dim()
    ret = []
    for i in range(self.dim()):
        if i < rstd_dim:
            ret.append(self.size(i))
        else:
            ret.append(1)
    rstd = torch.empty(ret, dtype=torch.float32, device='meta')
    return (torch.empty_like(rstd), torch.empty_like(rstd), torch.empty_like(self, dtype=self.dtype))


@impl(m, "npu_rms_norm")
def npu_rms_norm_meta(self, gamma, epsilon=1e-6):
    rstd_dim = self.dim() - gamma.dim()
    ret = []
    for i in range(self.dim()):
        if i < rstd_dim:
            ret.append(self.size(i))
        else:
            ret.append(1)
    rstd = torch.empty(ret, dtype=torch.float32, device='meta')
    return (torch.empty_like(self, dtype=self.dtype), torch.empty_like(rstd))


@impl(m, "npu_add_rms_norm")
def npu_add_rms_norm_meta(x1, x2, gamma, epsilon=1e-6):
    rstd_dim = x1.dim() - gamma.dim()
    ret = []
    for i in range(x1.dim()):
        if i < rstd_dim:
            ret.append(x1.size(i))
        else:
            ret.append(1)
    rstd = torch.empty(ret, dtype=torch.float32, device='meta')
    return (torch.empty_like(x1, dtype=x1.dtype), torch.empty_like(rstd), torch.empty_like(x1, dtype=x1.dtype))


@impl(m, "npu_add_rms_norm_cast")
def npu_add_rms_norm_cast_meta(x1, x2, gamma, epsilon=1e-6):
    rstd_dim = x1.dim() - gamma.dim()
    ret = []
    for i in range(x1.dim()):
        if i < rstd_dim:
            ret.append(x1.size(i))
        else:
            ret.append(1)
    rstd = torch.empty(ret, dtype=torch.float32, device='meta')
    return (torch.empty_like(x1, dtype=torch.float32), torch.empty_like(x1, dtype=x1.dtype), torch.empty_like(rstd), torch.empty_like(x1, dtype=x1.dtype))


@impl(m, "npu_rms_norm_backward")
def npu_rms_norm_backward_meta(dy, self, gamma, rstd):
    return (torch.empty_like(self, dtype=self.dtype), torch.empty_like(gamma, dtype=gamma.dtype))


@impl(m, "scatter_update")
def scatter_update_meta(self, indices, updates, axis):
    return torch.empty_like(self)


@impl(m, "scatter_update_")
def scatter_update__meta(self, indices, updates, axis):
    return self


@impl(m, "_npu_dropout")
def _npu_dropout_meta(self, p):
    mask = math.floor(math.floor((self.numel() + BIT_NUMBER - 1) / BIT_NUMBER) * BIT_NUMBER / UINT8_BIT_NUMBER)
    return (torch.empty_like(self, dtype=self.dtype), torch.empty(mask, dtype=torch.uint8, device='meta'))


@impl(m, "npu_quant_scatter")
def npu_quant_scatter_meta(self, indices, updates, quant_scales, quant_zero_points=None, axis=0, quant_axis=1,
                           reduce='update'):
    return torch.empty_like(self)


@impl(m, "npu_quant_scatter_")
def npu_quant_scatter__meta(self, indices, updates, quant_scales, quant_zero_points=None, axis=0, quant_axis=1,
                            reduce='update'):
    return self


@impl(m, "npu_scatter_list_")
def scatter_list__meta(self, indices, updates, mask, reduce='update', axis=-2):
    return self


@impl(m, "npu_scatter_list")
def scatter_list_meta(self, indices, updates, mask, reduce='update', axis=-2):
    var_list = []
    for item in self:
        var_list.append(torch.empty_like(item))
    return var_list


@impl(m, "npu_scatter_nd_update")
def scatter_nd_update_meta(self, indices, updates):
    return torch.empty_like(self, dtype=self.dtype)


@impl(m, "npu_scatter_nd_update_")
def scatter_nd_update__meta(self, indices, updates):
    return self


if torch.__version__ >= '2.3.1':
    @impl(m, "npu_geglu")
    def npu_geglu_meta(self, dim, approximate, activate_left=False):
        return (torch.empty_like(self, dtype=self.dtype), torch.empty_like(self, dtype=self.dtype))


    @impl(m, "npu_geglu_grad")
    def npu_geglu_backward_meta(grad_output, self, gelu, dim, approximate, activate_left=False):
        return (torch.empty_like(self, dtype=self.dtype), torch.empty_like(self, dtype=self.dtype))


@impl(m, "npu_dropout_backward")
def npu_dropout_backward_meta(grad_output, mask, p):
    return torch.empty_like(grad_output, dtype=grad_output.dtype)


@impl(m, "npu_masked_softmax_with_rel_pos_bias")
def npu_masked_softmax_with_rel_pos_bias_meta(x, atten_mask, relative_pos_bias, scale_value=1.0, inner_precision_mode=0):
    return torch.empty_like(x, dtype=x.dtype)


@impl(m, "npu_moe_distribute_dispatch")
def npu_moe_distribute_dispatch_meta(x, expert_ids, group_ep, ep_world_size, ep_rank_id, moe_expert_num, scales=None, x_active_mask=None, expert_scales=None, group_tp="", tp_world_size=0,
                                     tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, quant_mode=0, global_bs=0, expert_token_nums_type=1):
    n = x.size(0)
    h = x.size(1)
    k = expert_ids.size(1)

    shared_front = 0
    outDtype = x.dtype
    if expert_shard_type == 0:
        shared_front = 1

    local_moe_expert_num = 0
    global_bs_real = 0
    if global_bs == 0:
        global_bs_real = n * ep_world_size
    else:
        global_bs_real = global_bs
    a = 0
    if shared_front == 1:
        if ep_rank_id < shared_expert_rank_num:
            local_moe_expert_num = 1
            a = global_bs_real // shared_expert_rank_num
        else:
            local_moe_expert_num = moe_expert_num // (ep_world_size - shared_expert_rank_num)
            a = global_bs_real * min(local_moe_expert_num, k)
    else:
        if ep_rank_id >= ep_world_size - shared_expert_rank_num:
            local_moe_expert_num = 1
            a = global_bs_real // shared_expert_rank_num
        else:
            local_moe_expert_num = moe_expert_num // (ep_world_size - shared_expert_rank_num)
            a = global_bs_real * min(local_moe_expert_num, k)
    ep_recv_cnt_num = 0
    if tp_world_size == 2:
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num * tp_world_size
    else:
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num

    if scales is not None or quant_mode != 0:
        outDtype = torch.int8
    local_moe_expert_num = int(local_moe_expert_num)

    expand_idx = x.new_empty(tuple([n * k]), dtype=torch.int32)
    if tp_world_size == 0:
        expand_x = x.new_empty(tuple([a, h]), dtype=outDtype)
        dynamic_scales = x.new_empty(tuple([a]), dtype=torch.float32)
    else:
        expand_x = x.new_empty(tuple([a * tp_world_size, h]), dtype=outDtype)
        dynamic_scales = x.new_empty(tuple([a * tp_world_size]), dtype=torch.float32)
    expert_token_nums = x.new_empty(tuple([local_moe_expert_num]), dtype=torch.int64)
    ep_recv_counts = x.new_empty(tuple([ep_recv_cnt_num]), dtype=torch.int32)
    tp_recv_counts = x.new_empty(tuple([tp_world_size]), dtype=torch.int32)
    expand_scales = x.new_empty(tuple([0]), dtype=torch.float32)
    if expert_scales is not None:
        ep_recv_cnt_num = ep_world_size * local_moe_expert_num + n * 2 * k * (ep_world_size / 8)
        ep_recv_counts = x.new_empty(tuple([ep_recv_cnt_num]), dtype=torch.int32)
        expand_scales = x.new_empty(tuple([a]), dtype=torch.float32)
    return (expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales)


@impl(m, "npu_moe_distribute_combine")
def npu_moe_distribute_combine_meta(expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num,
                                    tp_send_counts=None, x_active_mask=None, activation_scale=None, weight_scale=None, group_list=None, expand_scales=None, group_tp="", tp_world_size=0,
                                    tp_rank_id=0, expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0, global_bs=0, out_dtype=0, comm_quant_mode=0, group_list_type=0):
    dim_list = []
    dim_list.append(expert_ids.size(0))
    dim_list.append(expand_x.size(1))

    return expand_x.new_empty(tuple(dim_list), dtype=expand_x.dtype)


@impl(m, "npu_ffn")
def npu_ffn_meta(x, weight1, weight2, activation, *, expert_tokens=None, expert_tokens_index=None, bias1=None,
                 bias2=None, scale=None, offset=None, deq_scale1=None, deq_scale2=None, antiquant_scale1=None,
                 antiquant_scale2=None, antiquant_offset1=None, antiquant_offset2=None, inner_precise=0,
                 output_dtype=None):
    dim_list = []
    for i in range(0, x.dim() - 1):
        dim_list.append(x.size(i))
    dim_list.append(weight2.size(weight2.dim() - 1))
    if x.dtype == torch.int8:
        if output_dtype is not None and output_dtype == torch.bfloat16:
            return x.new_empty(tuple(dim_list), dtype=torch.bfloat16)
        else:
            return x.new_empty(tuple(dim_list), dtype=torch.float16)
    else:
        return x.new_empty(tuple(dim_list))


@impl(m, "npu_grouped_matmul")
@impl(m, "npu_grouped_matmul.List")
def npu_grouped_matmul_meta(x, weight, *, bias=None, scale=None, offset=None, antiquant_scale=None,
                            antiquant_offset=None, per_token_scale=None, group_list=None,
                            activation_input=None, activation_quant_scale=None, activation_quant_offset=None,
                            split_item=0, group_type=-1, group_list_type=0, act_type=0, output_dtype=None):
    y = []
    num_x = len(x)
    singleWeight = len(weight) == 1 and len(weight[0].shape) == 3
    n = weight[0].shape[2] if singleWeight else weight[0].shape[1]
    if num_x > 0 and output_dtype is None:
        output_dtype = x[0].dtype
    if split_item == 0:
        for i in range(num_x):
            ni = n if singleWeight else weight[i].shape[1]
            y.append(x[i].new_empty((*x[i].shape[:-1], ni), dtype=output_dtype))
    elif split_item == 1:
        num_group_list = group_list.shape[0] if isinstance(group_list, torch.Tensor) else len(group_list)
        pre_offset = group_list[0]
        y.append(x[0].new_empty((pre_offset, n), dtype=output_dtype))
        for i in range(1, num_group_list):
            ni = n if singleWeight else weight[i].shape[1]
            cur_offset = group_list[i]
            y.append(x[0].new_empty((cur_offset - pre_offset, ni), dtype=output_dtype))
            pre_offset = cur_offset
    elif split_item == 2:
        dim_m = 0
        for i in range(num_x):
            dim_m += x[i].shape[0]
        y.append(x[0].new_empty((dim_m, n), dtype=output_dtype))
    elif split_item == 3:
        y.append(x[0].new_empty((x[0].shape[0], n), dtype=output_dtype))

    return y


@impl(m, "npu_grouped_matmul_finalize_routing")
def npu_grouped_matmul_finalize_routing_meta(x, w, group_list, *, scale=None, bias=None, pertoken_scale=None,
                                            shared_input=None, logit=None, row_index=None, dtype=None,
                                            shared_input_weight=1.0, shared_input_offset=0, output_bs=0,
                                            group_list_type=1):
    dimm = x.size(0)
    x_dim = x.dim()
    w_dim = w.dim()
    dimn = w.size(w_dim - 1)

    torch._check(
        x_dim == 2 and w_dim == 3,
        lambda: "input tensor only support shared_input and logit empty tensor" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        dimn > 0,
        lambda: "n value must bigger than 0." + ops_error(ErrCode.VALUE),
    )

    scene1 = False
    scene2 = False
    scene1 = (scale is not None and pertoken_scale is not None and
              group_list is not None and shared_input is not None and
              logit is not None and row_index is not None)
    scene2 = (scale is not None and pertoken_scale is not None and
              group_list is not None and shared_input is None and
              logit is None and row_index is not None)
    torch._check(
        scene1 or scene2,
        lambda: "input tensor only support shared_input and logit empty tensor" + ops_error(ErrCode.VALUE),
    )
    if dtype is None:
        dtype = torch.float32
    if shared_input is not None and logit is not None:
        torch._check(
            dtype == torch.float32,
            lambda: "When shared_input is not None, output_dtype must be float32, but it is " +
                    str(dtype) + ops_error(ErrCode.TYPE),
        )

    y_dimm = output_bs
    if output_bs == 0:
        y_dimm = dimm

    dim_list = [y_dimm, dimn]

    if dtype == torch.float32:
        return x.new_empty(tuple(dim_list), dtype=torch.float32)
    else:
        raise RuntimeError("Not supportted output dtype is " + str(dtype))


@impl(m, "npu_group_norm_silu")
def group_norm_silu_meta(self, gemma, beta, group, eps=0.00001):
    N = self.size(1)
    if gemma is None or beta is None:
        return (torch.empty_like(self, dtype=self.dtype), self.new_empty((N, group), dtype=self.dtype), self.new_empty((N, group), dtype=self.dtype))
    else:
        return (torch.empty_like(self, dtype=self.dtype), gemma.new_empty((N, group), dtype=gemma.dtype), beta.new_empty((N, group), dtype=beta.dtype))


@impl(m, "npu_mm_all_reduce_base")
def npu_mm_all_reduce_base_forward(x1, x2, hcom, reduce_op='sum', bias=None, antiquant_scale=None,
                                   antiquant_offset=None, x3=None, dequant_scale=None, pertoken_scale=None,
                                   comm_quant_scale_1=None, comm_quant_scale_2=None, antiquant_group_size=0,
                                   comm_turn=0):
    dim_list = []
    for i in range(x1.dim()):
        dim_list.append(x1.size(i))
    dim_list[-1] = x2.size(1)
    if dequant_scale is not None:
        if dequant_scale.dtype == torch.bfloat16:
            return x1.new_empty(tuple(dim_list), dtype=torch.bfloat16)
        else:
            return x1.new_empty(tuple(dim_list), dtype=torch.float16)
    else:
        return x1.new_empty(tuple(dim_list))



@impl(m, "npu_weight_quant_batchmatmul")
def npu_weight_quant_batchmatmul_meta(x, weight, antiquant_scale, antiquant_offset=None, quant_scale=None, quant_offset=None, bias=None, antiquant_group_size=0, inner_precise=0):
    dim_m = x.size(0)
    if weight.dtype == torch.int32 and weight.is_contiguous():
        dim_n = weight.size(1) * 8
    else:
        dim_n = weight.size(1)
    if quant_scale is not None:
        return x.new_empty((dim_m, dim_n), dtype=torch.int8)
    return x.new_empty((dim_m, dim_n), dtype=x.dtype)


def bias_shape_check(x2, bias, batch_val, is_a4w4, transpose_x2):
    bias_dim_num = bias.dim()
    if is_a4w4:
        torch._check(
            bias_dim_num == 1,
            lambda: "bias_dim_num should be 1 when x1's dtype is int32, please check bias dim num " + ops_error(ErrCode.VALUE),
        )
    else:
        torch._check(
            bias_dim_num == 1 or bias_dim_num == 3,
            lambda: "bias_dim_num should be 1 or 3 when x1's dtype is int8, please check bias dim num " + ops_error(ErrCode.VALUE),
        )
    x2_dim_num = x2.dim()
    x2_n_dim = x2.size(x2_dim_num - 1) * 8 if (is_a4w4 and not transpose_x2) else x2.size(x2_dim_num - 1)
    bias_first_dim = bias.size(0)
    if bias_dim_num == 1:
        torch._check(
            bias_first_dim == x2_n_dim,
            lambda: "bias_first_dim should be equal to x2 n dim, please check bias 1st dim value " + ops_error(ErrCode.VALUE),
        )
        return
    bias_second_dim = bias.size(1)
    bias_third_dim = bias.size(2)
    torch._check(
        bias_first_dim == batch_val,
        lambda: "infered batch value should be equal to bias batch dim value, please check bias batch dim value" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        bias_second_dim == 1,
        lambda: "bias_second_dim should be 1, please check bias second dim value " + ops_error(ErrCode.VALUE),
    )
    torch._check(
        bias_third_dim == x2_n_dim,
        lambda: "bias_third_dim should be equal to x2_n_dim, please check bias third dim value " + ops_error(ErrCode.VALUE),
    )


def quant_matmul_shape_check(*args):
    x1, x2, scale, offset, pertoken_scale, is_a4w4, transpose_x2 = args
    X_MAX_DIM = 6
    X_MIN_DIM = 2
    INT4_IN_INT32 = 8
    x1_dim_num = x1.dim()
    x2_dim_num = x2.dim()
    x1_m_dim = x1.size(x1_dim_num - 2)
    x1_k_dim = x1.size(x1_dim_num - 1)
    x2_k_dim = x2.size(x2_dim_num - 2)
    x2_n_dim = x2.size(x2_dim_num - 1) * INT4_IN_INT32 if (is_a4w4 and not transpose_x2) else x2.size(x2_dim_num - 1)
    torch._check(
        x1_dim_num >= X_MIN_DIM and x1_dim_num <= X_MAX_DIM,
        lambda: "x1 dim num should be 2 ~ 6, please check x1 dim num" + ops_error(ErrCode.VALUE),
    )
    if is_a4w4 and not transpose_x2:
        torch._check(
            x1_k_dim * INT4_IN_INT32 == x2_k_dim,
            lambda: "k dim of x2 should be 8 multiple of k dim of x1, please check k dim of x1 and x2" + ops_error(ErrCode.VALUE),
        )
    else:
        torch._check(
            x1_k_dim == x2_k_dim,
            lambda: "k dim of x1 and x2 need be same, please check k dim of x1 and x2" + ops_error(ErrCode.VALUE),
        )

    if is_a4w4:
        torch._check(
            x2_dim_num == X_MIN_DIM,
            lambda: "x2 dim num should be 2 when x1's dtype is int32, please check x2 dim num" + ops_error(ErrCode.VALUE),
        )
    else:
        torch._check(
            x2_dim_num >= X_MIN_DIM and x2_dim_num <= X_MAX_DIM,
            lambda: "x2 dim num should be 2 ~ 6 when x1's dtype is int8, please check x2 dim num" + ops_error(ErrCode.VALUE),
        )

    if offset is not None:
        offset_dim_num = offset.dim()
        torch._check(
            offset_dim_num == 1,
            lambda: "the offset dim num must be 1, please check offset dim num " + ops_error(ErrCode.VALUE),
        )
        offset_first_dim = offset.size(0)
        torch._check(
            offset_first_dim == 1 or offset_first_dim == x2_n_dim,
            lambda: "the offset 1st dim value must be 1 or x2 n dim value, please check offset 1st dim value" + ops_error(ErrCode.VALUE),
        )
    if pertoken_scale is not None:
        pertoken_scale_dim_num = pertoken_scale.dim()
        torch._check(
            pertoken_scale_dim_num == 1,
            lambda: "the pertoken_scale dim num must be 1, please check scale dim num" + ops_error(ErrCode.VALUE),
        )
        pertoken_scale_first_dim = pertoken_scale.size(0)
        torch._check(
            pertoken_scale_first_dim == x1_m_dim,
            lambda: "the pertoken_scale 1st dim value must be x1 m dim value, please check scale 1st dim value " + ops_error(ErrCode.VALUE),
        )

    scale_dim_num = scale.dim()
    torch._check(
        scale_dim_num == 1,
        lambda: "the scale dim num must be 1, please check scale dim num" + ops_error(ErrCode.VALUE),
    )
    scale_first_dim = scale.size(0)
    torch._check(
        scale_first_dim == 1 or scale_first_dim == x2_n_dim,
        lambda: "the scale 1st dim value must be 1 or x2 n dim value, please check scale 1st dim value " + ops_error(ErrCode.VALUE),
    )


def quant_matmul_bias_dtype_check(bias, pertoken_scale, output_dtype):
    bias_dtype_supported_list = [torch.int32, torch.bfloat16, torch.float32, torch.float16]
    torch._check(
        bias.dtype in bias_dtype_supported_list,
        lambda: "bias's type supported for int32, bfloat16, float16 and float32, but bias.dtype is " + str(bias.dtype) + ops_error(ErrCode.TYPE),
    )
    if bias.dtype == torch.bfloat16:
        torch._check(
            output_dtype == torch.bfloat16,
            lambda: "When bias dtype is bfloat16, output_dtype must be bfloat16, but it is " +
                    str(output_dtype) + ops_error(ErrCode.TYPE),
        )
    if output_dtype == torch.int32:
        torch._check(
            bias.dtype == torch.int32,
            lambda: "When output_dtype dtype is int32, bias_dtype must be int32, but it is " +
                    str(bias.dtype) + ops_error(ErrCode.TYPE),
        )
    if pertoken_scale is not None:
        if bias.dtype == torch.float16:
            torch._check(
                output_dtype == torch.float16,
                lambda: "When bias dtype is float16 and pertoken is given, output_dtype must be float16, but it is " +
                        str(output_dtype) + ops_error(ErrCode.TYPE),
            )
    else:
        torch._check(
            bias.dtype != torch.float16,
            lambda: "Bias dtype cannot be float16 when pertoken not given." + ops_error(ErrCode.TYPE),
        )
        if bias.dtype == torch.float32:
            torch._check(
                output_dtype == torch.bfloat16,
                lambda: "When bias dtype is float32 and pertoken not given, output_dtype must be bfloat16, but it is " +
                        str(output_dtype) + ops_error(ErrCode.TYPE),
            )


def quant_matmul_dtype_check(*args):
    x1, x2, scale, offset, pertoken_scale, bias, output_dtype, is_a4w4 = args
    torch._check(
        x1.dtype == x2.dtype,
        lambda: "x1's type and x2's type should be same, but x1.dtype is " + str(x1.dtype) + " and x2.dtype is " +
                str(x2.dtype) + ops_error(ErrCode.TYPE),
    )
    input_dtype_supported_list = [torch.int8, torch.int32]
    torch._check(
        x1.dtype in input_dtype_supported_list,
        lambda: "input's type supported for int8 and int32, but now is " + str(x1.dtype) + ops_error(ErrCode.TYPE),
    )
    scale_dtype_supported_list = [torch.float32, torch.int64, torch.bfloat16]
    torch._check(
        scale.dtype in scale_dtype_supported_list,
        lambda: "scale's type supported for float32, int64 and bfloat16, but scale.dtype is " + str(scale.dtype) + ops_error(ErrCode.TYPE),
    )
    if offset is not None:
        torch._check(
            offset.dtype == torch.float32,
            lambda: "offset's type supported for float32, but offset.dtype is " + str(offset.dtype) + ops_error(ErrCode.TYPE),
        )
    if pertoken_scale is not None:
        torch._check(
            pertoken_scale.dtype == torch.float32,
            lambda: "pertoken_scale's type supported for float32, but pertoken_scale.dtype is " +
                    str(offset.dtype) + ops_error(ErrCode.TYPE),
        )
    if bias is not None:
        quant_matmul_bias_dtype_check(bias, pertoken_scale, output_dtype)


def quant_matmul_scale_offset_out_check(scale, offset, pertoken_scale, output_dtype, is_a4w4):
    if scale.dtype == torch.bfloat16:
        torch._check(
            output_dtype in [torch.bfloat16, torch.int32],
            lambda: "When scale's dtype is bfloat16, output_dtype must be bfloat16 or int32, but output_dtype is " +
                    str(output_dtype) + ops_error(ErrCode.TYPE),
        )
    if output_dtype == torch.bfloat16:
        torch._check(
            scale.dtype == torch.bfloat16 or scale.dtype == torch.float32,
            lambda: "When output_dtype is bfloat16, scale's dtype must be bfloat16 or float32, but scale's dtype is " +
                    str(scale.dtype) + ops_error(ErrCode.TYPE),
        )
    if output_dtype == torch.int32:
        torch._check(
            scale.dtype in [torch.bfloat16, torch.float32],
            lambda: "When output_dtype is int32, scale's dtype must be bfloat16 or float32, but scale's dtype is " +
                    str(scale.dtype) + ops_error(ErrCode.TYPE),
        )
    if offset is not None:
        torch._check(
            output_dtype is None or output_dtype == torch.int8,
            lambda: "offset only exists when output_dtype is int8, but output_dtype is " + str(output_dtype) + ops_error(ErrCode.TYPE),
        )
    if pertoken_scale is not None:
        if output_dtype == torch.float16:
            torch._check(
                scale.dtype == torch.float32,
                lambda: "When output_dtype is float16 and pertoken_scale is not none, scale's dtype must be float32, but scale's dtype is " +
                        str(scale.dtype) + ops_error(ErrCode.TYPE),
            )
        torch._check(
            output_dtype == torch.float16 or output_dtype == torch.bfloat16,
            lambda: "When pertoken_scale is not none, output_dtype must be float16 or bfloat16, but output_dtype is " +
                    str(output_dtype) + ops_error(ErrCode.TYPE),
        )
    if is_a4w4 and pertoken_scale is None:
        torch._check(
            output_dtype == torch.float16,
            lambda: "When input's dtype is int32, output_dtype must be float16, but output_dtype is " +
                    str(output_dtype) + ops_error(ErrCode.TYPE),
        )


@impl(m, "npu_quant_matmul")
def npu_quant_matmul_meta(x1, x2, scale, *, offset=None, pertoken_scale=None, bias=None, output_dtype=None):
    INT4_IN_INT32 = 8
    batch_val = 1
    x1_dim_num = x1.dim()
    x2_dim_num = x2.dim()
    out_dim_num = max(x1_dim_num, x2_dim_num)
    shape_long = x1 if x1_dim_num > x2_dim_num else x2
    shape_short = x2 if x1_dim_num > x2_dim_num else x1
    vaild_offset = out_dim_num - min(x1_dim_num, x2_dim_num)
    is_a4w4 = x1.dtype == torch.int32 and x2.dtype == torch.int32
    dim_list = []
    for i in range(0, out_dim_num - 2):
        short_dim = 1 if i < vaild_offset else shape_short.size(i - vaild_offset)
        long_dim = shape_long.size(i)
        torch._check(
            not (short_dim > 1 and long_dim > 1 and short_dim != long_dim),
            lambda: "the batch shape cannot be broadcast" + ops_error(ErrCode.VALUE),
        )
        cur_batch_val = max(short_dim, long_dim)
        batch_val = batch_val * cur_batch_val
        dim_list.append(cur_batch_val)
    dimm = x1.size(x1.dim() - 2)
    transpose_x2 = x1.size(x1.dim() - 1) == x2.size(x2.dim() - 2)

    dimn = x2.size(x2.dim() - 1) * INT4_IN_INT32 if (is_a4w4 and not transpose_x2) else x2.size(x2.dim() - 1)
    dim_list.append(dimm)
    dim_list.append(dimn)
    quant_matmul_shape_check(x1, x2, scale, offset, pertoken_scale, is_a4w4, transpose_x2)
    if bias is not None:
        if bias.dim() == 3:
            torch._check(
                len(dim_list) == 3,
                lambda:"when bias dim is 3, out dim need to be 3" + ops_error(ErrCode.TYPE),
            )
        bias_shape_check(x2, bias, batch_val, is_a4w4, transpose_x2)
    quant_matmul_dtype_check(x1, x2, scale, offset, pertoken_scale, bias, output_dtype, is_a4w4)
    quant_matmul_scale_offset_out_check(scale, offset, pertoken_scale, output_dtype, is_a4w4)
    if output_dtype == torch.float16:
        return shape_long.new_empty(tuple(dim_list), dtype=torch.float16)
    elif output_dtype == torch.bfloat16:
        return shape_long.new_empty(tuple(dim_list), dtype=torch.bfloat16)
    elif output_dtype == torch.int32:
        return shape_long.new_empty(tuple(dim_list), dtype=torch.int32)
    elif output_dtype is None or output_dtype == torch.int8:
        return shape_long.new_empty(tuple(dim_list), dtype=torch.int8)
    else:
        raise RuntimeError("Not supportted output dtype is " + str(output_dtype))


@impl(m, "npu_quant_matmul_dequant")
def npu_quant_matmul_dequant_meta(x, quantized_weight, weight_scale, *,
                                  bias=None, x_scale=None, x_offset=None, smooth_scale=None, quant_mode="pertoken"):
    torch._check(
        x.dim() == 2,
        lambda: "the x dim support only 2" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        x.dtype == torch.float16,
        lambda: "the x dtype support only float16" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        quantized_weight.dim() == 2,
        lambda: "the quantized_weight dim support only 2" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        quantized_weight.dtype == torch.int8,
        lambda: "the quantized_weight dtype support only int8" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        weight_scale.dim() == 1,
        lambda: "the weight_scale dim support only 1" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        weight_scale.dtype == torch.float,
        lambda: "the weight_scale dtype support only float" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        x.shape[1] == quantized_weight.shape[1],
        lambda: "x shape[1] not equal to quantized_weight shape[1]" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        weight_scale.shape[0] == quantized_weight.shape[0],
        lambda: "weight_scale shape[0] not equal to quantized_weight shape[0]" + ops_error(ErrCode.VALUE),
    )
    return torch.empty((x.shape[0], weight_scale.shape[0]), dtype=x.dtype, device='meta')


@impl(m, "npu_quant_grouped_matmul_dequant")
def npu_quant_grouped_matmul_dequant_meta(x, quantized_weight, weight_scale, group_list, *,
                                          bias=None, x_scale=None, x_offset=None, smooth_scale=None, quant_mode="pertoken"):
    torch._check(
        x.dim() == 2,
        lambda: "the x dim support only 2" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        x.dtype == torch.float16,
        lambda: "the x dtype support only float16" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        quantized_weight.dim() == 3,
        lambda: "the quantized_weight dim support only 3" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        quantized_weight.dtype == torch.int8,
        lambda: "the quantized_weight dtype support only int8" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        weight_scale.dim() == 2,
        lambda: "the weight_scale dim support only 2" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        weight_scale.dtype == torch.float,
        lambda: "the weight_scale dtype support only float" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        x.shape[1] == quantized_weight.shape[2],
        lambda: "x shape[1] not equal to quantized_weight shape[1]" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        weight_scale.shape[0] == quantized_weight.shape[0],
        lambda: "weight_scale shape[0] not equal to quantized_weight shape[0]" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        weight_scale.shape[1] == quantized_weight.shape[1],
        lambda: "weight_scale shape[1] not equal to quantized_weight shape[1]" + ops_error(ErrCode.VALUE),
    )
    return torch.empty((x.shape[0], weight_scale.shape[1]), dtype=x.dtype, device='meta')


@impl(m, "npu_trans_quant_param")
def npu_trans_quant_param_meta(scale, offset=None, round_mode=0):
    scale_dim_num = scale.dim()
    torch._check(
        scale_dim_num == 1 or (scale_dim_num == 2 and scale.size(0) == 1),
        lambda: "the scale shape support only (1, ) and (1, n)" + ops_error(ErrCode.VALUE),
    )
    torch._check(
        round_mode == 0 or round_mode == 1,
        lambda: "round_mode should be 0 or 1, but round_mode is " + int(round_mode) + ops_error(ErrCode.VALUE),
    )
    output_shape = scale.size()
    if scale_dim_num == 1:
        scale_first_dim = scale.size(0)
        dim_max = scale_first_dim
        if offset is not None:
            offset_first_dim = offset.size(0)
            dim_max = max(dim_max, offset_first_dim)
            if offset_first_dim != 1 and scale_first_dim != 1:
                torch._check(
                    offset_first_dim == scale_first_dim,
                    lambda: "offset first dim should be equal to scale first dim if none of them are equal to one" + ops_error(ErrCode.VALUE),
                )
        output_shape = (dim_max)
    else:
        if offset is not None:
            torch._check(
                scale.size() == offset.size(),
                lambda: "when the input shape of scale is (1, n), shape of scale and offset should be equal" + ops_error(ErrCode.VALUE),
            )
    return scale.new_empty(output_shape, dtype=torch.int64)


@impl(m, "npu_quantize")
def npu_quantize_meta(self, scales, zero_points, dtype, axis=1, div_mode=True):
    if dtype == torch.quint8:
        return torch.empty_like(self, dtype=torch.uint8)
    elif dtype == torch.qint8:
        return torch.empty_like(self, dtype=torch.int8)
    elif dtype == torch.qint32:
        return torch.empty_like(self, dtype=torch.int32)
    elif dtype == torch.quint4x2:
        dim_num = self.dim()
        if self.size(dim_num - 1) % 8:
            raise RuntimeError("If dtype is quint4x2, last dim must be divided by 8" +
                               ops_error(ErrCode.NOT_SUPPORT))
        output_shape = []
        for dim in range(dim_num - 1):
            output_shape.append(self.size(dim))
        output_shape.append(self.size(dim_num - 1) // 8)
        return self.new_empty(output_shape, dtype=torch.int32)
    return torch.empty_like(self, dtype=torch.int8)


@impl(m, "npu_group_quant")
def npu_group_quant_meta(x, scale, group_index, *, offset=None, dst_dtype=None):
    if dst_dtype == torch.quint8:
        return torch.empty_like(x, dtype=torch.uint8)
    elif dst_dtype == torch.qint8:
        return torch.empty_like(x, dtype=torch.int8)
    elif dst_dtype == torch.quint4x2:
        dim_num = x.dim()
        if x.size(dim_num - 1) % 8:
            raise RuntimeError("If dst_dtype is quint4x2, last dim must be divisible by 8" +
                               ops_error(ErrCode.NOT_SUPPORT))
        output_shape = []
        for dim in range(dim_num - 1):
            output_shape.append(x.size(dim))
        output_shape.append(x.size(dim_num - 1) // 8)
        return x.new_empty(output_shape, dtype=torch.int32)
    return torch.empty_like(x, dtype=torch.int8)


@impl(m, "npu_dynamic_quant")
def npu_dynamic_quant(input_dummy, *, smooth_scales=None, group_index=None, dst_type=torch.int8):
    dim_num = input_dummy.dim()
    scale_shape = []
    for dim in range(dim_num - 1):
        scale_shape.append(input_dummy.size(dim))
    scale = input_dummy.new_empty(scale_shape, dtype=torch.float32)
    if dst_type == torch.quint4x2:
        if input_dummy.size(dim_num - 1) % 8:
            raise RuntimeError("If dst_dtype is quint4x2, last dim must be divisible by 8" +
                               ops_error(ErrCode.PARAM))
        scale_shape.append(input_dummy.size(dim_num - 1) // 8)
        output = input_dummy.new_empty(scale_shape, dtype=torch.int32)
    else:
        output = torch.empty_like(input_dummy, dtype=torch.int8)
    return (output, scale)


@impl(m, "npu_dynamic_quant_asymmetric")
def npu_dynamic_quant_asymmetric(input_dummy, *, smooth_scales=None, group_index=None, dst_type=torch.int8):
    dim_num = input_dummy.dim()
    scale_offset_shape = []
    for dim in range(dim_num - 1):
        scale_offset_shape.append(input_dummy.size(dim))
    scale = input_dummy.new_empty(scale_offset_shape, dtype=torch.float32)
    offset = input_dummy.new_empty(scale_offset_shape, dtype=torch.float32)
    if dst_type == torch.quint4x2:
        if input_dummy.size(dim_num - 1) % 8:
            raise RuntimeError("If dst_dtype is quint4x2, last dim must be divisible by 8" +
                               ops_error(ErrCode.PARAM))
        scale_offset_shape.append(input_dummy.size(dim_num - 1) // 8)
        output = input_dummy.new_empty(scale_offset_shape, dtype=torch.int32)
    else:
        output = torch.empty_like(input_dummy, dtype=torch.int8)
    return (output, scale, offset)


@impl(m, "npu_moe_compute_expert_tokens")
def npu_moe_compute_expert_tokens_meta(sorted_experts, num_experts=1):
    out = torch.zeros(num_experts, dtype=torch.int32, device='meta')
    return torch.empty_like(out)


@impl(m, "npu_anti_quant")
def npu_anti_quant_meta(x, scale, *, offset=None, dst_dtype=None, src_dtype=None):
    if dst_dtype is None:
        dst_dtype = torch.float16

    if x.dtype == torch.int32:
        x_shape = x.size()
        if len(x_shape) == 0:
            raise RuntimeError("Not supported for x is scalar when x dtype is int32" + ops_error(ErrCode.NOT_SUPPORT))
        y_shape = (*(x_shape[:-1]), x_shape[-1] * 8)
        y = x.new_empty(y_shape, dtype=dst_dtype)
        return torch.empty_like(y)
    else:
        return torch.empty_like(x, dtype=dst_dtype)


@impl(m, "npu_kronecker_quant")
def npu_kronecker_quant_meta(x, kronecker_p1, kronecker_p2, clip_ratio=1.0, dst_dtype=None):
    dim_num = x.dim()
    if x.size(dim_num - 1) % 8:
        raise RuntimeError("last dim of input x must be divisible by 8" + ops_error(ErrCode.NOT_SUPPORT))
    output_shape = []
    for dim in range(dim_num - 1):
        output_shape.append(x.size(dim))
    output_shape.append(x.size(dim_num - 1) // 8)

    scale_shape = []
    scale_shape.append(x.size(0))
    return x.new_empty(output_shape, dtype=torch.int32), x.new_empty(scale_shape, dtype=torch.float32)


@impl(m, "npu_kv_rmsnorm_rope_cache")
def npu_kv_rmsnorm_rope_cache_meta(kv, gamma, cos, sin, index, k_cache, ckv_cache, *, k_rope_scale=None,
                                   c_kv_scale=None, k_rope_offset=None, c_kv_offset=None, epsilon=1e-5,
                                   cache_mode='Norm', is_output_kv=False):
    if kv.dim() != 4:
        raise RuntimeError("4D tensor expected for input kv" + ops_error(ErrCode.PARAM))
    if gamma.dim() != 1:
        raise RuntimeError("1D tensor expected for input gamma" + ops_error(ErrCode.PARAM))
    if cos.dim() != 4:
        raise RuntimeError("4D tensor expected for input cos" + ops_error(ErrCode.PARAM))
    k_rope_size = []
    c_kv_size = []
    for i in range(kv.dim() - 1):
        k_rope_size.append(kv.size(i))
        c_kv_size.append(kv.size(i))
    k_rope_size.append(cos.size(3))
    c_kv_size.append(gamma.size(0))
    return (torch.empty_like(k_cache), torch.empty_like(ckv_cache),
            torch.empty(k_rope_size, dtype=kv.dtype, device=kv.device),
            torch.empty(c_kv_size, dtype=kv.dtype, device=kv.device))


@impl(m, "npu_apply_rotary_pos_emb")
def npu_apply_rotary_pos_emb_meta(query, key, cos, sin, layout=1):
    return (torch.empty_like(query, dtype=query.dtype), torch.empty_like(key, dtype=key.dtype))


@impl(m, "npu_quant_conv2d")
def npu_quant_conv2d(input_, weight, scale, strides, pads, dilations,
                     groups=1, offset_x=0, round_mode='rint', output_dtype=None, bias=None, offset=None):

    input_shape = input_.size()
    weight_shape = weight.size()
    scale_shape = scale.size()

    input_dim = input_.dim()
    weight_dim = weight.dim()
    scale_dim = scale.dim()

    def check_basic_inputs_dim_shape():

        torch._check(
            input_dim == weight_dim and weight_dim == INPUTS_DIM_LIMIT_QUANTCONV2D,
            lambda: "input dim or weight dim is not equal to 4, but now input dim is " + str(input_dim) + ", and weight dim is "
                     + str(weight_dim) + ops_error(ErrCode.VALUE),
        )

        torch._check(
            scale_dim == 1,
            lambda: "scale dim is not equal to 1, but now scale dim is " + str(scale_dim) + ops_error(ErrCode.VALUE),
        )

        torch._check(
            input_shape[1] == weight_shape[1],
            lambda: "input cin should equal to weight cin, but now input cin is " + str(input_shape[1]) + ", and weight cin is "
                    + str(weight_shape[1]) + ops_error(ErrCode.VALUE),
        )

        torch._check(
            scale_shape[0] == weight_shape[0],
            lambda: "scale shape should equal to cout, but now scale shape is " + str(scale_shape[0]) + ", and cout is " +
                    str(weight_shape[0]) + ops_error(ErrCode.VALUE),
        )

    def check_basic_inputs_dtype():
        torch._check(
            input_.dtype == torch.int8 and weight.dtype == torch.int8,
            lambda: "input's dtype and weight's dtype should be int8, but input.dtype is " + str(input_.dtype) + ", and weight.dtype is " +
                    str(weight.dtype) + ops_error(ErrCode.TYPE),
        )

        torch._check(
            scale.dtype == torch.int64,
            lambda: "scale's dtype should be int64, but scale.dtype is " + str(scale.dtype) + ops_error(ErrCode.TYPE),
        )

        torch._check(
            output_dtype == torch.float16,
            lambda: "output dtype should be float16, but now dtype is " + str(output_dtype) + ops_error(ErrCode.TYPE),
        )

    def check_bias_dim_shape_dtype():
        bias_dim = bias.dim()
        bias_shape = bias.size()
        torch._check(
            bias_dim == 1,
            lambda: "bias dim is not equal to 1, but now bias dim is " + str(bias_dim) + ops_error(ErrCode.VALUE),
        )

        torch._check(
            bias.dtype == torch.int32,
            lambda: "bias' dtype should be int32, but bias.dtype is " + str(input_.dtype) + ops_error(ErrCode.VALUE),
        )

        torch._check(
            bias_shape[0] == weight_shape[0],
            lambda: "bias shape should equal to cout, but now bias shape is " + str(bias_shape[0]) + ", and cout is " +
                    str(weight_shape[0]) + ops_error(ErrCode.VALUE),
        )

    def check_attrs():
        pads_dim = len(pads)
        strides_dim = len(strides)
        dilations_dim = len(dilations)
        torch._check(
            pads_dim == ATTR_DIM_LIMIT_QUANTCONV2D and strides_dim == ATTR_DIM_LIMIT_QUANTCONV2D and
            dilations_dim == ATTR_DIM_LIMIT_QUANTCONV2D,
            lambda: "attrs's dim should be 2, but pads dim is " + str(pads_dim) + ", strides dim is "
                    + str(strides_dim) + ", dilations dim is " + str(dilations_dim) + ops_error(ErrCode.VALUE),
        )
        torch._check(
            pads[0] >= 0 and pads[1] >= 0,
            lambda: "pads's value should large or equal to 0, but pads is " + str(pads[0]) + ", "
                    + str(pads[1]) + ops_error(ErrCode.VALUE),
        )
        torch._check(
            strides[0] > 0 and strides[1] > 0,
            lambda: "strides's value should large than 0, but strides is " + str(strides[0]) + ", "
                    + str(strides[1]) + ops_error(ErrCode.VALUE),
        )
        torch._check(
            dilations[0] > 0 and dilations[1] > 0,
            lambda: "dilations's value should large than 0, but dilations is " + str(dilations[0]) + ", "
                    + str(dilations[1]) + ops_error(ErrCode.VALUE),
        )
        torch._check(
            groups == 1,
            lambda: "groups should be 1, but now " + str(groups) + ops_error(ErrCode.VALUE),
        )
        torch._check(
            offset_x <= 127 and offset_x >= -128,
            lambda: "offset_x should be [-128,127], but offset_x is " + str(offset_x) + ops_error(ErrCode.VALUE),
        )
        torch._check(
            round_mode == 'rint',
            lambda: "round_mode should be rint, but round_mode is " + str(round_mode) + ops_error(ErrCode.VALUE),
        )

    check_basic_inputs_dim_shape()
    check_basic_inputs_dtype()
    if bias is not None:
        check_bias_dim_shape_dtype()
    check_attrs()

    nout = input_shape[0]
    cout = weight_shape[0]
    hout = (input_shape[2] + pads[0] * 2 - dilations[0] * (weight_shape[2] - 1) - 1) // strides[0] + 1
    wout = (input_shape[3] + pads[1] * 2 - dilations[1] * (weight_shape[3] - 1) - 1) // strides[1] + 1

    torch._check(
        hout > 0 and wout > 0,
        lambda: "ho, wo should larger than 0, but now ho is " + str(hout) + ", and wo is " + str(wout) + ops_error(ErrCode.VALUE),
    )

    output_dim_list = [nout, cout, hout, wout]

    return scale.new_empty(tuple(output_dim_list), dtype=output_dtype)


@impl(m, "npu_linear")
def npu_linear_meta(input_, weight, bias=None):
    dimm = input_.size(0)
    dimn = weight.size(0)
    return input_.new_empty((dimm, dimn))


@impl(m, "npu_moe_finalize_routing")
def npu_moe_finalize_routing_meta(expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row,
                                  expert_for_source_row, drop_pad_mode=0):
    if scales is None:
        return torch.empty_like(expanded_permuted_rows, dtype=expanded_permuted_rows.dtype)
    dimm = scales.size(0)
    if drop_pad_mode == 1 or drop_pad_mode == 3:
        # dropPad场景, expanded_permuted_rows shape 为[E, C, H]
        dimn = expanded_permuted_rows.size(2)
    else:
        # dropLess场景, expanded_permuted_rows shape 为[NUM_ROWS * K, H]
        dimn = expanded_permuted_rows.size(1)
    return expanded_permuted_rows.new_empty((dimm, dimn))


has_side_effect(torch.ops.npu.npu_prefetch.default)


@impl(m, "npu_prefetch")
def npu_prefetch_meta(self, dependency, max_size, offset=0):
    torch._check(
        max_size > 0,
        lambda: f"The max_size should be greater than zero, but got {max_size}.",
    )
    torch._check(
        offset >= 0,
        lambda: f"The offset should be nonnegative, but got {offset}.",
    )


@impl(m, "npu_swiglu")
def npu_swiglu_meta(x, dim=-1):
    output_size = []
    for i in range(x.dim()):
        output_size.append(x.size(i))
    output_size[dim] = math.floor(output_size[dim] / 2)
    return torch.empty(output_size, dtype=x.dtype, device=x.device)


@impl(m, "npu_swiglu_backward")
def npu_swiglugrad_meta(y, x, dim=-1):
    return torch.empty_like(x)


def rope_quant_kvcache(x, cos, k_cache, v_cache, size_splits, kv_output=False):
    torch._check(
        x.dim() == 3 or x.dim() == 2,
        lambda: f"The x's dim should be 2 or 3, but got {x.dim()}.",
        )
    torch._check(
        k_cache.dim() == 4,
        lambda: f"The k_cache's dim should be 4, but got {k_cache.dim()}.",
        )
    num_size_splits = len(size_splits)
    torch._check(
        num_size_splits == 3,
        lambda: f"The size_splits should be 3, but got {num_size_splits}.",
        )
    torch._check(
        size_splits[0] >= 0,
        lambda: f"size_splits[0] should not less than 0, but got {size_splits[0]}.",
        )
    batch = x.size(0)
    seqlen = x.size(1)
    k_headdim = k_cache.size(2)
    hidden_size = k_cache.size(3)
    q_headdim = 0
    if hidden_size != 0:
        q_headdim = size_splits[0] // hidden_size
    out_q_size = [batch, seqlen, q_headdim, hidden_size] if x.dim() == 3 else [batch, q_headdim, hidden_size]
    out_k_size = [0]
    out_v_size = [0]
    if kv_output:
        out_k_size = [batch, seqlen, k_headdim, hidden_size] if x.dim() == 3 else [batch, k_headdim, hidden_size]
        out_v_size = [batch, seqlen, k_headdim, hidden_size] if x.dim() == 3 else [batch, k_headdim, hidden_size]
    return (torch.empty(out_q_size, dtype=cos.dtype, device=x.device),
            torch.empty(out_k_size, dtype=cos.dtype, device=x.device),
            torch.empty(out_v_size, dtype=cos.dtype, device=x.device),
            k_cache, v_cache)


@impl(m, "npu_dequant_swiglu_quant")
def npu_dequant_swiglu_quant_meta(x, weight_scale=None, activation_scale=None, bias=None, quant_scale=None,
                                  quant_offset=None, group_index=None, activate_left=False, quant_mode=0):
    y_size = []
    scale_size = []
    for i in range(x.dim() - 1):
        y_size.append(x.size(i))
        scale_size.append(x.size(i))
    y_size.append(math.floor(x.size(x.dim() - 1) / 2))
    return (torch.empty(y_size, dtype=torch.int8, device=x.device),
            torch.empty(scale_size, dtype=torch.float32, device=x.device))


@impl(m, "npu_dequant_rope_quant_kvcache")
def npu_dequant_rope_quant_kvcache_meta(x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, size_splits, *,
                                        offset_k=None, offset_v=None, weight_scale=None, activation_scale=None,
                                        bias=None, quant_mode=0, input_layout="BSND", kv_output=False,
                                        cache_mode="contiguous"):
    torch._check(
        x.dtype == torch.int32,
        lambda: f"The x's dtype should be Int32, but got {x.dtype}.",
        )
    return rope_quant_kvcache(x, cos, k_cache, v_cache, size_splits, kv_output=kv_output)


@impl(m, "npu_rope_quant_kvcache")
def npu_rope_quant_kvcache_meta(x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, size_splits, *, offset_k=None,
                                offset_v=None, quant_mode=0, input_layout="BSND", kv_output=False, cache_mode="contiguous"):
    return rope_quant_kvcache(x, cos, k_cache, v_cache, size_splits, kv_output=kv_output)


@impl(m, "npu_dequant_bias")
def npu_dequant_bias_meta(x, weight_scale, activation_scale, bias, output_dtype=None):
    if output_dtype is None:
        output_dtype = torch.float16
    if output_dtype != torch.float16 and output_dtype != torch.bfloat16:
        raise RuntimeError("Only supported output_dtype is float16 and bfloat16" + ops_error(ErrCode.NOT_SUPPORT))
    return torch.empty_like(x, dtype=output_dtype)


@impl(m, "npu_interleave_rope")
def npu_interleave_rope_meta(x, cos, sin):
    return torch.empty_like(x)


@impl(m, "npu_batch_gather_matmul")
def npu_batch_gather_matmul_meta(self, x, weight_b, indices, weight_a=None,
                                 layer_idx=0, scale=1e-3, y_offset=0, y_slice_size=-1):
    return torch.empty_like(self, dtype=self.dtype)


@impl(m, "npu_batch_gather_matmul_")
def npu_batch_gather_matmul__meta(self, x, weight_b, indices, weight_a=None,
                                 layer_idx=0, scale=1e-3, y_offset=0, y_slice_size=-1):
    return self


@impl(m, "npu_gather_backward")
def npu_gather_backward__meta(grad, self_size, dim, index, sparse_grad):
    return torch.empty(self_size, dtype=grad.dtype, device=grad.device)


@impl(m, "npu_moe_re_routing")
def npu_moe_re_routing_meta(tokens, expert_token_num_per_rank, per_token_scales=None, expert_token_num_type=1, idx_type=0):
    permute_tokens_size = []
    permute_per_token_scales_size = []
    permute_token_idx_size = []
    expert_token_num_size = []
    for i in range(tokens.dim()):
        permute_tokens_size.append(tokens.size(i))
    permute_per_token_scales_size.append(tokens.size(0))
    permute_token_idx_size.append(tokens.size(0))
    expert_token_num_size.append(expert_token_num_per_rank.size(1))
    return (torch.empty(permute_tokens_size, dtype=tokens.dtype, device=tokens.device),
            torch.empty(permute_per_token_scales_size, dtype=torch.float32, device=tokens.device),
            torch.empty(permute_token_idx_size, dtype=torch.int32, device=tokens.device),
            torch.empty(expert_token_num_size, dtype=expert_token_num_per_rank.dtype, device=tokens.device))
