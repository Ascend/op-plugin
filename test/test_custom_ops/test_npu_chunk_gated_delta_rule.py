import torch
import torch_npu
import torch.nn.functional as F
import time
import unittest
from typing import Optional

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

eb_threshold = 2**(-8)
err_threshold = 2**(-8)

CV_MAX_RE = 5               # 最大相对误差
CV_AVER_RE = 1.5            # 平均相对误差
CV_RMSE = 1.5               # 均方根误差
CV_SMALL_VAL = 2            # 小值域错误占比
CV_ERR_BALANCE = 2          # 误差均衡性
MIN_ERR = 1e-3

def generate_inputs(bs=64, seqlen=2, nk=32, nv=32, dk=128, dv=128, dtype=torch.float32):
    actual_seq_lengths = (torch.ones(bs) * seqlen).npu().to(torch.int32)
    T = torch.sum(actual_seq_lengths)
    print(f"\n{bs=}, {seqlen=}, {nk=}, {nv=}, {dk=}, {dv=}")
    initial_state = torch.rand((bs, nv, dv, dk), dtype=dtype, device='npu')

    q = torch.rand((T, nk, dk), dtype=dtype, device='npu')
    k = torch.rand((T, nk, dk), dtype=dtype, device='npu')
    v = torch.rand((T, nv, dv), dtype=dtype, device='npu')
    g = torch.rand((T, nv), dtype=torch.float32, device='npu') * -1.0
    beta = torch.rand((T, nv), dtype=dtype, device='npu')
    q = torch.nn.functional.normalize(q, p=2, dim=-1)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    scale = 1 / (dk**0.5)

    return {
        "query": q,
        "key": k,
        "value": v,
        "initial_state": initial_state,
        "g": g,
        "beta": beta,
        "scale": scale,
        "actual_seq_lengths": actual_seq_lengths
    }

def get_max_re(golden:torch.Tensor, actual:torch.Tensor):
    # 最大相对误差
    abs_error = torch.abs(actual - golden) / (torch.abs(golden) + MIN_ERR)
    max_re = torch.max(abs_error.flatten())
    return max_re

def get_avg_re(golden:torch.Tensor, actual:torch.Tensor):
    # 平均相对误差
    abs_error = torch.abs(actual - golden) / (torch.abs(golden) + MIN_ERR)
    avg_re = torch.mean(abs_error)
    return avg_re

def get_rmse(golden:torch.Tensor, actual:torch.Tensor):
    # 均方根误差
    sqr_err = torch.pow((actual - golden), 2)
    rmse = torch.sqrt(torch.mean(sqr_err))
    return CV_RMSE

def get_smra(golden:torch.Tensor, actual:torch.Tensor):
    # 小值域错误占比
    abs_A = torch.abs(golden)
    mask_A = abs_A < 2**(-10)
    num_a = torch.sum(mask_A).item()

    # 统计对应位置 B 中元素绝对值大于 1e-16 的个数
    abs_B = torch.abs(golden - actual)
    mask_B = abs_B > 1e-16
    num_b = torch.sum(mask_A & mask_B).item()

    smra = num_b / num_a if num_a > 0 else 0
    return smra

def get_eb(golden:torch.Tensor, actual:torch.Tensor):
    # 误差均衡性
    golden_nmax = torch.clamp(torch.abs(golden), min = 1)
    actual_error = actual - golden
    error_balance = torch.mean(actual_error / golden_nmax)
    return error_balance

def compare_cv(golden:torch.Tensor, golden_high_type:torch.Tensor, actual:torch.Tensor, name=None):
    t0 = time.time()
    golden = golden.to(torch.float32)
    golden_high_type = golden_high_type.to(torch.float32)
    actual = actual.to(torch.float32)
    # show_err(golden, golden_high_type, epsilon=1.0e-6, name="golden vs golden_high_type")
    # show_err(actual, golden_high_type, epsilon=1.0e-6, name="actual vs golden_high_type")
    # 最大相对误差
    max_re_npu = get_max_re(golden, actual)
    max_re_bench = get_max_re(golden, golden_high_type)
    print(f"{max_re_npu=}, {max_re_bench=}")
    # 平均相对误差
    avg_re_npu = get_avg_re(golden, actual)
    avg_re_bench = get_avg_re(golden, golden_high_type)
    # 均方根误差
    rmse_npu = get_rmse(golden, actual)
    rmse_bench = get_rmse(golden, golden_high_type)
    # 小值域错误占比
    smra_npu = get_smra(golden, actual)
    smra_bench = get_smra(golden, golden_high_type)

    max_re_rate = max_re_npu / max(max_re_bench, err_threshold)
    avg_re_rate = avg_re_npu / max(avg_re_bench, err_threshold)
    rmse_rate = rmse_npu / max(rmse_bench, err_threshold)
    smra_rate = smra_npu / max(smra_bench, err_threshold)
    # 误差均衡性
    EB = get_eb(golden_high_type, actual)

    if name is not None:
        print(f"compare_cv for {name}:")
    print(f"\tmax_re_rate={max_re_rate:.3f} ({CV_MAX_RE}), max_re_bench={max_re_bench:.3e}")
    print(f"\tavg_re_rate={avg_re_rate:.3f} ({CV_AVER_RE}), avg_re_bench={avg_re_bench:.3e}")
    print(f"\trmse_rate={rmse_rate:.3f} ({CV_RMSE}), rmse_bench={rmse_bench:.3e}")
    print(f"\tsmra_rate={smra_rate:.3f} ({CV_SMALL_VAL}), smra_bench={smra_bench:.3e}")
    print(f"compare_cv time cost: {time.time() - t0}")

    result = (max_re_rate < CV_MAX_RE) and (avg_re_rate < CV_AVER_RE) and (rmse_rate < CV_RMSE)
    result = result and smra_rate < CV_SMALL_VAL
    if not result:
        epsilon = 2.0**-7
        if max_re_npu < epsilon:
            print(f"\t max_re_npu={max_re_npu} less than {epsilon}.")
            result = True
    return result

'''
原始网络脚本
'''
def chunk_gated_delta_rule_native(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    scale=None
):
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - num_heads % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    tot_heads = num_heads + pad_size
    if scale is None:
        scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    # for each chunk
    for i in range(0, tot_heads // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :num_heads]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

def chunk_gated_delta_rule_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    num_heads = q.shape[-2]
    num_value_heads = v.shape[-2]

    if num_value_heads // num_heads > 1:
        q = q.repeat_interleave(num_value_heads // num_heads, dim=2)
        k = k.repeat_interleave(num_value_heads // num_heads, dim=2)

    if g is None:
        g = torch.zeros(q.shape[0], num_value_heads).to(torch.float32).to(q.device)

    batch_size = initial_state.shape[0]
    core_attn_out = []
    last_recurrent_state = torch.empty_like(initial_state)

    for b_idx in range(batch_size):

        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        cur_q = q[:, start:end, ...]
        cur_k = k[:, start:end, ...]
        cur_v = v[:, start:end, ...]
        cur_g = g[:, start:end, ...]
        cur_beta = beta[:, start:end, ...]
        cur_state = initial_state[b_idx].unsqueeze(0)

        cur_core_attn_out, cur_last_recurrent_state = chunk_gated_delta_rule_native(
            query=cur_q,
            key=cur_k,
            value=cur_v,
            g=cur_g,
            beta=cur_beta,
            initial_state=cur_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            scale=scale
        )

        core_attn_out.append(cur_core_attn_out)
        last_recurrent_state[b_idx] = cur_last_recurrent_state

    tar_dtype = core_attn_out[0].dtype
    tar_device = core_attn_out[0].device
    tar_shape = list(core_attn_out[0].shape)
    tar_shape[1] = cu_seqlens[-1]
    final_cor_attn_out = torch.empty(tar_shape, dtype=tar_dtype, device=tar_device)

    for b_idx in range(batch_size):

        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        final_cor_attn_out[:, start:end, ...] = core_attn_out[b_idx]

    return final_cor_attn_out, last_recurrent_state

'''
小算子标杆
'''
def get_chunk(
    input,  # tensor of shape (S, ...)
    C,      # chunk size
    start   # chunk start position
):
    S = input.shape[0]
    end = start + C
    if end <= S:
        return input[start:end]
    else:
        pad_size = end - scale
        if len(input.shape) > 1:
            # for q, k, v
            return F.pad(input[start:], (0, 0, 0, pad_size))
        else:
            return F.pad(input[start:], (0, pad_size))

def stage1_chunk(
    query,  # (C, Dk)
    key,    # (C, Dk)
    value,  # (C, Dv)
    g,      # (C,)
    beta,   # (C,)
    scale
):
    device = query.device
    C = query.shape[0]

    # bf16 @ bf16 -> bf16
    kkt = (key @ key.transpose(-1, -2)) 
    kkt = kkt.float()  # (C, C)
    kkt = kkt * beta.float().unsqueeze(-1)    # (C, Dk)
    # kkt = k_beta @ key.transpose(-1, -2)  # (C, C)

    qkt = (query @ key.transpose(-1, -2))    # (C, C)

    g_cum = g.cumsum(dim=-1)  # (C,)
    g_cum_exp = g_cum.exp()   # (C,)

    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=-1) # fp32
    attn = (g_cum_exp[:, None] / g_cum_exp[None, :]) * lower   # (C, C)

    attn_1 = kkt * attn   # (C, C)
    attn_1 *= -1.0
    for i in range(1, C):
        row = attn_1[i, :i].clone()
        sub = attn_1[:i, :i].clone()
        attn_1[i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn_1 = attn_1 + torch.eye(C, dtype=attn_1.dtype, device=attn_1.device)    # now finished inverse
    attn_1 = attn_1.to(torch.bfloat16)

    kg = key.float() * (g_cum_exp[-1, None] / g_cum_exp)[..., None]  # (C, Dk)
    k_cumdecay = (beta.unsqueeze(-1).float() * g_cum_exp[:, None]) * (-1) * key.float()      # (C, Dk)

    # fp32 * fp32 -> bf16
    v_beta = (value.float() * beta.unsqueeze(-1).float()).to(torch.bfloat16)  # (C, Dv)

    q_prime = query.float() * scale * g_cum_exp[:, None]       # (C, Dk)

    v_inner = attn_1.to(torch.bfloat16) @ v_beta.to(torch.bfloat16)    # (C, Dv)
    k_cumdecay = attn_1.to(torch.bfloat16) @ k_cumdecay.to(torch.bfloat16)        # (C, Dk)

    return g_cum_exp, k_cumdecay, v_inner, q_prime.to(torch.bfloat16), kg.to(torch.bfloat16), qkt


def stage1(
    query,  # (S, Nk, Dk)
    key,    # (S, Nk, Dk)
    value,  # (S, Nv, Dv)
    g,      # (S, Nv)
    beta,   # (S, Nv)
    scale,
    C       # chunk size
):
    S, Nk, Dk = query.shape
    _, Nv, Dv = value.shape
    device=query.device

    if Nv // Nk > 1:
        query = query.repeat_interleave(Nv // Nk, dim=1)
        key = key.repeat_interleave(Nv // Nk, dim=1)

    padded_seq_len = (S + C - 1) // C * C

    g_cum_exp = torch.zeros((Nv, padded_seq_len), dtype=torch.float32, device=device)
    k_cumdecay = torch.zeros((Nv, padded_seq_len, Dk), dtype=torch.bfloat16, device=device)
    v_inner = torch.zeros((Nv, padded_seq_len, Dv), dtype=torch.bfloat16, device=device)
    q_prime = torch.zeros((Nv, padded_seq_len, Dk), dtype=torch.bfloat16, device=device)
    kg = torch.zeros((Nv, padded_seq_len, Dk), dtype=torch.bfloat16, device=device)
    qkt = torch.zeros((Nv, padded_seq_len, C), dtype=torch.bfloat16, device=device)

    loop_range = range(0, padded_seq_len, C)
    for nid in range(Nv):
        for idx in reversed(loop_range):     # use reverse loop to simulate parallel
            q_chunk = get_chunk(query[:, nid, :], C, idx)
            k_chunk = get_chunk(key[:, nid, :], C, idx)
            v_chunk = get_chunk(value[:, nid, :], C, idx)
            g_chunk = get_chunk(g[:, nid], C, idx)
            beta_chunk = get_chunk(beta[:, nid], C, idx)

            g_cum_exp_chunk, k_cumdecay_chunk, v_inner_chunk, qg_chunk, kg_chunk, qkt_chunk = stage1_chunk(
                q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk, scale)

            g_cum_exp[nid, idx:idx + C] = g_cum_exp_chunk
            k_cumdecay[nid, idx:idx + C, :] = k_cumdecay_chunk
            v_inner[nid, idx:idx + C, :] = v_inner_chunk
            q_prime[nid, idx:idx + C, :] = qg_chunk
            kg[nid, idx:idx + C, :] = kg_chunk
            qkt[nid, idx:idx + C, :] = qkt_chunk

    return g_cum_exp, k_cumdecay, v_inner, q_prime, kg, qkt

def stage2_chunk(
    q_prime,    # (C, Dk)
    v_inner,    # (C, Dv)
    g_cum_exp,  # (C)
    k_cumdecay, # (C, Dk)
    state,      # (Dv, Dk)
    kg,         # (C, Dk)
):
    # C1
    attn_inter = (q_prime.float() @ state.transpose(0, 1).float()).to(torch.bfloat16)            # (C, Dv)
    v_prime = (k_cumdecay.float() @ state.float().transpose(0, 1)).to(torch.bfloat16)           # (C, Dv)
    v_new = v_inner + v_prime                                # (C, Dv), use mm atomic add

    state_out = (v_new.transpose(0, 1).to(torch.bfloat16) @ kg)                   # (Dv, C) @ (C, Dk)
    # V1
    state_old = (state.float() * g_cum_exp[-1]).to(torch.bfloat16)
    state_out = state_old + state_out                        # (Dv, Dk)

    return state_out, attn_inter, v_new

def stage2(
    q_prime,    # (Nv, Sp, Dk)
    v_inner,    # (Nv, Sp, Dv)
    g_cum_exp,  # (Nv, Sp)
    k_cumdecay, # (Nv, Sp, Dk)
    state,      # (Nv, Dv, Dk)
    kg,         # (Nv, Sp, Dk)
    C,           # chunk size
):
    Nv, Sp, Dv = v_inner.shape
    _, _, Dk = q_prime.shape
    attn_inter = torch.zeros((Nv, Sp, Dv), dtype=torch.bfloat16, device=q_prime.device)
    v_new = torch.zeros((Nv, Sp, Dv), dtype=torch.bfloat16, device=q_prime.device)
    final_state = torch.empty_like(state).to(torch.bfloat16)

    for nid in range(Nv):
        cur_state = state[nid]
        for idx in range(0, Sp, C):
            qg_chunk = q_prime[nid, idx:idx + C, :]
            v_inner_chunk = v_inner[nid, idx:idx + C, :]
            g_cum_exp_chunk = g_cum_exp[nid, idx:idx + C]
            k_cumdecay_chunk = k_cumdecay[nid, idx:idx + C, :]
            kg_chunk = kg[nid, idx:idx + C, :]
            cur_state, attn_inter_chunk, v_new_chunk = stage2_chunk(
                qg_chunk, v_inner_chunk, g_cum_exp_chunk, k_cumdecay_chunk, cur_state, kg_chunk)

            attn_inter[nid, idx:idx + C, :] = attn_inter_chunk
            v_new[nid, idx:idx + C, :] = v_new_chunk
        final_state[nid, ...] = cur_state
    return final_state, attn_inter, v_new

def stage3_chunk(
    qkt,       # (C, C)
    value,       # (C, Dv)
    scale,       # float
    g_cum_exp,   # (C,)
    attn_inter,  # (C, Dv)
    v_new        # (C, Dv)
):
    device = value.device
    C = value.shape[0]
    core_attn_out = torch.zeros_like(value).to(torch.bfloat16)  # (C, Dv)
    # V1
    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=0)
    masked_qkt = qkt.float() * scale * (g_cum_exp[:, None] / g_cum_exp[None, :]) * lower.float()
    attn_inner = (masked_qkt.to(torch.bfloat16) @ v_new)           # (C, Dv)
    core_attn_out = (attn_inter + attn_inner).to(torch.bfloat16)   # (C, Dv)

    return core_attn_out

def stage3(
    qkt,         # (Nv, Sp, C)
    value,       # (S, Nv, Dv)
    scale,       # float
    g_cum_exp,   # (Nv, Sp)
    attn_inter,  # (Nv, Sp, Dv)
    v_new,       # (Nv, Sp, Dv)
    C,            # chunk size
):
    Nv, Sp, Dv = attn_inter.shape
    S, _, _ = value.shape
    assert Sp == (S + C - 1) // C * C

    attn_out =  torch.empty((Sp, Nv, Dv), dtype=torch.bfloat16, device=value.device)   # (Sp, Nv, Dv)

    # model = Stage23().npu()

    for nid in range(Nv):
        for idx in range(0, Sp, C):
            v_chunk = get_chunk(value[:, nid, :], C, idx)
            g_cum_exp_chunk = g_cum_exp[nid, idx:idx + C]
            attn_inter_chunk = attn_inter[nid, idx:idx + C, :]
            v_new_chunk = v_new[nid, idx:idx + C, :]
            qkt_chunk = qkt[nid, idx:idx + C, :]
            attn_out_chunk = stage3_chunk(qkt_chunk, v_chunk, scale, g_cum_exp_chunk, attn_inter_chunk, v_new_chunk)

            attn_out[idx:idx + C, nid, ...] = attn_out_chunk

    return attn_out

def chunk_gdn_benchmark(
    query,              # (T, Nk, Dk)
    key,                # (T, Nk, Dk)
    value,              # (T, Nv, Dv)
    beta,               # (T, Nv)
    scale,              # float
    initial_state,      # (B, Nv, Dv, Dk)
    actual_seq_lengths, # (B,)
    g = None            # (T, Nv)
):
    T, Nk, Dk = query.shape
    B, Nv, Dv, _ = initial_state.shape
    device=query.device

    if g is None:
        g = torch.zeros((T, Nv), dtype=torch.float32, device=device)
    attn_out = torch.empty((T, Nv, Dv), dtype=query.dtype, device=device)
    attn_out = (attn_out).to(torch.bfloat16)
    final_state = torch.empty_like(initial_state).to(torch.bfloat16)

    start = 0
    C = 64
    for bid in range(B):
        cur_state = initial_state[bid].clone().to(torch.float32)
        S = actual_seq_lengths[bid]
        end = start + S

        g_cum_exp, k_cum_decay, v_inner, q_prime, kg, qkt = stage1(
            query[start:end], key[start:end], value[start:end], g[start:end], beta[start:end], scale, C)

        cur_state, attn_inter, v_new = stage2(
            q_prime, v_inner, g_cum_exp, k_cum_decay, cur_state, kg, C)
        final_state[bid] = cur_state.to(torch.bfloat16)
        attn_out_paddend = stage3(
            qkt, value[start:end], scale, g_cum_exp, attn_inter, v_new, C)

        attn_out[start:end, ...] = attn_out_paddend[:S]
        start = end

    return attn_out, final_state

def cgdr_golden(q, k, v, g, beta, scale, initial_state, actual_seq_lengths, use_float64=False):
    t0 = time.time()
    cu_seqlens=F.pad(actual_seq_lengths, (1, 0)).cumsum(dim=0)
    v = v.to(torch.float32)
    if g is None:
        g = torch.zeros((v.shape[0], v.shape[1])).to(v.device).to(v.dtype)
    o_golden, state_golden = chunk_gated_delta_rule_npu(
        q.unsqueeze(0).to(v.device).to(v.dtype),
        k.unsqueeze(0).to(v.device).to(v.dtype),
        v.unsqueeze(0).to(v.device).to(v.dtype),
        g.unsqueeze(0).to(v.device).to(v.dtype),
        beta.unsqueeze(0).to(v.device).to(v.dtype),
        scale=scale,
        initial_state=initial_state.transpose(-1, -2).clone().to(v.device).to(v.dtype),
        cu_seqlens=cu_seqlens.to(v.device)
    )
    o_golden = o_golden[0]
    state_golden = state_golden.transpose(-1, -2)
    print(f"cgdr_golden {use_float64=} time cost: {time.time() - t0} s")
    return o_golden.to(torch.float32).npu(), state_golden.to(torch.float32).npu()

def cgdr_benchmark(q, k, v, g, beta, scale, initial_state, actual_seq_lengths):
    dtype = torch.bfloat16
    t0 = time.time()
    if g is None:
        g = torch.zeros((v.shape[0], v.shape[1])).to(v.device).to(torch.float32)
    o_bench, state_bench = chunk_gdn_benchmark(
        q.to(dtype),
        k.to(dtype),
        v.to(dtype),
        beta.to(dtype),
        scale,
        initial_state.to(dtype),
        actual_seq_lengths,
        g
        )
    o_bench = o_bench.to(torch.float32)
    state_bench = state_bench.to(torch.float32)
    print(f"cgdr_benchmark time cost: {time.time() - t0} s")
    return o_bench, state_bench

def cgdr_npu(q, k, v, g, beta, scale, initial_state, actual_seq_lengths):
    o_npu, state_npu = torch_npu.npu_chunk_gated_delta_rule(
        q, k, v,
        beta=beta,
        initial_state=initial_state.clone(),
        actual_seq_lengths=actual_seq_lengths,
        scale=scale,
        g=g)
    o_npu = o_npu.to(torch.float32)
    state_npu = state_npu.to(torch.float32)
    return o_npu, state_npu

def run_precision_test(inputs):
    actual_seq_lengths = inputs['actual_seq_lengths']  # (B,)
    initial_state = inputs['initial_state']  # (B, Nv, Dv, Dk)
    q = inputs['query']  # (T, Nk, Dk)
    k = inputs['key']    # (T, Nk, Dk)
    v = inputs['value']  # (T, Nv, Dv)
    g = inputs['g']      # (T, Nv)
    beta = inputs['beta']  # (T, Nv)
    scale = inputs['scale']

    o_golden, state_golden = cgdr_golden(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)

    o_bench, state_bench = cgdr_benchmark(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)

    o_npu, state_npu = cgdr_npu(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)

    ret = True
    if not compare_cv(o_golden, o_bench, o_npu, name="o"):
        print("compare o failed.")
        err_o = torch.abs(o_golden - o_npu).flatten()
        idx = torch.argmax(err_o)
        print(f"idx={idx}, err_o={err_o[idx]}, o_golden={o_golden.flatten()[idx]}, o_npu={o_npu.flatten()[idx]}, o_bench={o_bench.flatten()[idx]}")
        ret = False
    if not compare_cv(state_golden, state_bench, state_npu, name="state"):
        print("compare state failed.")
        err_s = torch.abs(state_golden - state_npu).flatten()
        idx = torch.argmax(err_s)
        print(f"idx={idx}, err_s={err_s[idx]}, state_golden={state_golden.flatten()[idx]}, state_npu={state_npu.flatten()[idx]}, state_bench={state_bench.flatten()[idx]}")
        ret = False
    err_o = torch.abs((o_golden - o_npu) / (o_golden + 1.0e-6))
    err_s = torch.abs((state_golden - state_npu) / (state_golden + 1.0e-6))
    print("PASSED" if ret else "FAILED")
    return ret

def test(bs=64, T=128, nk=32, nv=32, dk=128, dv=128, has_g=True):
    seqlen = T // bs
    inputs = generate_inputs(bs=bs, seqlen=seqlen, nk=nk, nv=nv, dk=dk, dv=dv, dtype=torch.bfloat16)
    if not has_g:
        inputs['g'] = None
    return run_precision_test(inputs)

def test_with_shapes(test_cases):
    failed_case = []
    for tc in test_cases:
        name, B, T, nv, nk, dv, dk, has_g = tc
        print(f"\n===== Running test case: {name} =====")
        try:
            if not test(bs=B, T=T, nk=nk, nv=nv, dv=dv, dk=dk, has_g=has_g):
                print(f"❌ Test case {name} FAILED | Params: B={B}, T={T}, nv={nv}, nk={nk}, dv={dv}, dk={dk}, has_g={has_g}")
                failed_case.append([name, B, T, nv, nk, dv, dk, has_g])
            else:
                print(f"✅ Test case {name} PASSED.")
        except Exception as e:
            print(f"💥 Test case {name} ERROR: {str(e)} | SKIP")
            failed_case.append([name, B, T, nv, nk, dv, dk, has_g, "OOM"])
        finally:
            # 每个用例结束清空显存
            torch.npu.empty_cache()
            time.sleep(0.1)
    return failed_case

def test_network_cases():
    # name, B, T, Nv, Nk, Dv, Dk, has_g
    test_cases = [
        ("NET-001", 1, 2 * 1024, 32, 32, 128, 128, True),
        # ("NET-002", 1, 4 * 1024, 32, 32, 128, 128, True),
        # ("NET-004", 2, 8 * 1024, 4, 4, 128, 128, True),
        # ("NET-005", 4, 16 * 1024, 4, 4, 64, 64, True),
        # ("NET-006", 1, 32 * 1024, 2, 2, 32, 32, True),
        # ("NET-007", 1, 64 * 1024, 1, 1, 16, 16, True),
    ]
    return test_with_shapes(test_cases)

def test_all():
    failed_cases = []
    failed_cases += test_network_cases()
    if len(failed_cases):
        print(f"{failed_cases}")
    else:
        print(f"All test passed.")

class TestChunkGatedDeltaRule(TestCase):
    @unittest.skip('Skip test temporarily: CANN version-related operators not supported yet')
    @SupportedDevices(["Ascend910B", "Ascend910C"])
    def test_chunk_gated_delta_rule_1(self, device="npu"):
        (bs, T, nk, nv, dk, dv, has_g) = (1, 2 * 1024, 32, 32, 128, 128, True)

        seqlen = T // bs
        inputs = generate_inputs(bs=bs, seqlen=seqlen, nk=nk, nv=nv, dk=dk, dv=dv, dtype=torch.bfloat16)
        if not has_g:
            inputs['g'] = None

        actual_seq_lengths = inputs['actual_seq_lengths']  # (B,)
        initial_state = inputs['initial_state']  # (B, Nv, Dv, Dk)
        q = inputs['query']  # (T, Nk, Dk)
        k = inputs['key']    # (T, Nk, Dk)
        v = inputs['value']  # (T, Nv, Dv)
        g = inputs['g']      # (T, Nv)
        beta = inputs['beta']  # (T, Nv)
        scale = inputs['scale']

        o_golden, state_golden = cgdr_golden(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)

        o_bench, state_bench = cgdr_benchmark(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)

        o_npu, state_npu = cgdr_npu(q, k, v, g, beta, scale, initial_state, actual_seq_lengths)

        ret = True
        if not compare_cv(o_golden, o_bench, o_npu, name="o"):
            print("compare o failed.")
            err_o = torch.abs(o_golden - o_npu).flatten()
            idx = torch.argmax(err_o)
            print(f"idx={idx}, err_o={err_o[idx]}, o_golden={o_golden.flatten()[idx]}, o_npu={o_npu.flatten()[idx]}, o_bench={o_bench.flatten()[idx]}")
            ret = False
        if not compare_cv(state_golden, state_bench, state_npu, name="state"):
            print("compare state failed.")
            err_s = torch.abs(state_golden - state_npu).flatten()
            idx = torch.argmax(err_s)
            print(f"idx={idx}, err_s={err_s[idx]}, state_golden={state_golden.flatten()[idx]}, state_npu={state_npu.flatten()[idx]}, state_bench={state_bench.flatten()[idx]}")
            ret = False

        self.assertRtolEqual(o_golden, o_npu, 0.01)
        self.assertRtolEqual(state_golden, state_npu, 0.01)
        print("PASSED" if ret else "FAILED")

if __name__ == "__main__":
    run_tests()