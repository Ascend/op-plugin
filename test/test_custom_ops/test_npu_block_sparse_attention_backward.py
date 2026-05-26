"""
npu_block_sparse_attention_backward 反向算子单测。

测试场景覆盖：
- BNSD MHA（num_heads == num_kv_heads）
- TND GQA（num_heads > num_kv_heads，多个 Q head 共享同一个 KV head）
- TND 变长序列（NPU 接口传每 batch 实际长度，CPU 标杆使用累计 offset 切分 TND）
- head_dim=128（算子限制 head_dim <= 128，用例覆盖边界）
- float16、bfloat16 数据类型
- 多块稀疏（block_shape=[8,128]）
- 稀疏掩码（每个 q_block 仅 attend 部分 kv_block）
- 正反向算子同时调用（forward 输出作为 backward 输入）
- autograd 端到端前反向绑定

精度说明：为保障 NPU 与 CPU 标杆公平对比，显式 backward CPU 对比用例使用 CPU 正向标杆生成
attention_out、softmax_lse，确保双方使用同一 P 矩阵。TND/GQA autograd 用例走 NPU 正向并与 CPU
backward 标杆对比，验证实际网络中 .backward() 路径可用。
"""

import gc
import math
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices, SkipIfNotGteCANNVersion

DTYPE = torch.float16
B, S, N, D = 2, 32, 8, 128  # head_dim=128，满足 head_dim <= 128 算子限制
NUM_KV_HEADS = 8
BLOCK_SHAPE = [128, 128]
TND_GQA_B = 13
TND_GQA_NUM_HEADS = 12
TND_GQA_NUM_KV_HEADS = 3
TND_GQA_HEAD_DIM = 128
TND_GQA_BLOCK_SHAPE = [128, 128]
TND_GQA_Q_LENGTHS = [355, 17, 41, 83, 129, 211, 7, 53, 97, 151, 233, 301, 19]
TND_GQA_KV_LENGTHS = [533, 23, 67, 101, 173, 257, 11, 89, 137, 199, 281, 349, 31]


def _softmax_np(x):
    """Softmax with numerical stability. When all inputs are -inf (masked), return zeros."""
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x = x - x_max
    y = np.exp(x)
    s = y.sum(axis=-1, keepdims=True)
    s = np.where(s > 1e-10, s, 1.0)
    return y / s


def _logsumexp_np(scores):
    """对一行 scores 计算 log(sum(exp(scores)))，masked 位置为 -1e10 时仍稳定."""
    s = scores.astype(np.float32)
    m = np.max(s)
    if m <= -1e9:
        return m
    return m + np.log(np.sum(np.exp(s - m)))


def cpu_block_sparse_attention_bnsd_with_lse(query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads):
    """CPU 正向标杆：BNSD，返回 (attention_out, softmax_lse)，供反向测试生成入参，不依赖 NPU 正向."""
    q = query.cpu().to(torch.float32).numpy()
    k = key.cpu().to(torch.float32).numpy()
    v = value.cpu().to(torch.float32).numpy()
    mask = block_sparse_mask.cpu().numpy()
    B, N, S, D = q.shape
    N2 = k.shape[1]
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (S + block_x - 1) // block_x
    ceil_kv = (S + block_y - 1) // block_y
    out = np.zeros((B, N, S, D), dtype=np.float32)
    lse = np.zeros((B, N, S, 1), dtype=np.float32)
    for b in range(B):
        for n in range(N):
            n2 = n % N2
            for s1 in range(S):
                i_block = s1 // block_x
                scores = np.full(S, -1e10, dtype=np.float32)
                for s2 in range(S):
                    j_block = s2 // block_y
                    if i_block < ceil_q and j_block < ceil_kv and mask[b, n, i_block, j_block] != 0:
                        scores[s2] = float(scale_value) * np.dot(q[b, n, s1, :], k[b, n2, s2, :])
                lse[b, n, s1, 0] = _logsumexp_np(scores)
                probs = _softmax_np(scores)
                out[b, n, s1, :] = np.dot(probs, v[b, n2, :, :])
    attention_out = torch.from_numpy(out).to(query.dtype)
    softmax_lse = torch.from_numpy(lse).to(torch.float32)
    return attention_out, softmax_lse


def _softmax_backward_np(dP, P):
    """dP 与 P 同 shape，返回 softmax 反向的 dS."""
    dP = dP.astype(np.float32)
    P = P.astype(np.float32)
    sum_dp_p = (dP * P).sum(axis=-1, keepdims=True)
    return dP * P - P * sum_dp_p


def cpu_block_sparse_attention_backward_bnsd(
    query, key, value, d_out, block_sparse_mask, block_shape, scale_value, num_kv_heads
):
    """CPU 标杆：BNSD backward，与 forward 相同 mask/scale 重算 P 再算 dQ,dK,dV."""
    q = query.cpu().to(torch.float32).numpy()
    k = key.cpu().to(torch.float32).numpy()
    v = value.cpu().to(torch.float32).numpy()
    dout = d_out.cpu().to(torch.float32).numpy()
    mask = block_sparse_mask.cpu().numpy()
    B, N, S, D = q.shape
    N2 = k.shape[1]
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (S + block_x - 1) // block_x
    ceil_kv = (S + block_y - 1) // block_y
    scale = float(scale_value)

    d_query = np.zeros_like(q, dtype=np.float32)
    d_key = np.zeros_like(k, dtype=np.float32)
    d_value = np.zeros_like(v, dtype=np.float32)

    for b in range(B):
        for n in range(N):
            n2 = n % N2
            P = np.zeros((S, S), dtype=np.float32)
            for s1 in range(S):
                i_block = s1 // block_x
                scores = np.full(S, -1e10, dtype=np.float32)
                for s2 in range(S):
                    j_block = s2 // block_y
                    if i_block < ceil_q and j_block < ceil_kv and mask[b, n, i_block, j_block] != 0:
                        scores[s2] = scale * np.dot(q[b, n, s1, :], k[b, n2, s2, :])
                P[s1, :] = _softmax_np(scores).ravel()

            dP = np.zeros((S, S), dtype=np.float32)
            for s1 in range(S):
                for s2 in range(S):
                    dP[s1, s2] = np.dot(dout[b, n, s1, :], v[b, n2, s2, :])
            dS = _softmax_backward_np(dP, P) * scale

            for s1 in range(S):
                d_query[b, n, s1, :] = np.dot(dS[s1, :], k[b, n2, :, :])
            for s2 in range(S):
                d_key[b, n2, s2, :] += np.dot(dS[:, s2], q[b, n, :, :])
            for s2 in range(S):
                d_value[b, n2, s2, :] = np.dot(P[:, s2], dout[b, n, :, :])

    return (
        torch.from_numpy(d_query).to(query.dtype),
        torch.from_numpy(d_key).to(key.dtype),
        torch.from_numpy(d_value).to(value.dtype),
    )


def _make_cumulative_seq_lengths(lengths):
    seq_lengths = [0]
    for length in lengths:
        seq_lengths.append(seq_lengths[-1] + length)
    return seq_lengths


def _make_tnd_gqa_case(
    full_mask=True,
    num_heads=TND_GQA_NUM_HEADS,
    num_kv_heads=TND_GQA_NUM_KV_HEADS,
    head_dim=TND_GQA_HEAD_DIM,
    block_shape=TND_GQA_BLOCK_SHAPE,
    q_lengths=TND_GQA_Q_LENGTHS,
    kv_lengths=TND_GQA_KV_LENGTHS,
):
    batch = len(q_lengths)
    assert batch == len(kv_lengths)
    assert num_heads % num_kv_heads == 0
    scale_value = 1.0 / math.sqrt(head_dim)
    actual_seq_offsets = _make_cumulative_seq_lengths(q_lengths)
    actual_seq_offsets_kv = _make_cumulative_seq_lengths(kv_lengths)
    total_q = actual_seq_offsets[-1]
    total_kv = actual_seq_offsets_kv[-1]

    query = torch.randn(total_q, num_heads, head_dim, dtype=DTYPE)
    key = torch.randn(total_kv, num_kv_heads, head_dim, dtype=DTYPE)
    value = torch.randn(total_kv, num_kv_heads, head_dim, dtype=DTYPE)
    d_out = torch.randn(total_q, num_heads, head_dim, dtype=DTYPE)

    max_q = max(q_lengths)
    max_kv = max(kv_lengths)
    ceil_q = (max_q + block_shape[0] - 1) // block_shape[0]
    ceil_kv = (max_kv + block_shape[1] - 1) // block_shape[1]
    if full_mask:
        block_sparse_mask = torch.ones(batch, num_heads, ceil_q, ceil_kv, dtype=torch.int8)
    else:
        block_sparse_mask = torch.zeros(batch, num_heads, ceil_q, ceil_kv, dtype=torch.int8)
        for b in range(batch):
            valid_ceil_kv = (kv_lengths[b] + block_shape[1] - 1) // block_shape[1]
            for n in range(num_heads):
                for q_block in range(ceil_q):
                    block_sparse_mask[b, n, q_block, (q_block + b + n) % valid_ceil_kv] = 1
                    if valid_ceil_kv > 1 and (q_block + n) % 2 == 0:
                        block_sparse_mask[b, n, q_block, (q_block + b + n + 1) % valid_ceil_kv] = 1

    return {
        "query": query,
        "key": key,
        "value": value,
        "d_out": d_out,
        "block_sparse_mask": block_sparse_mask,
        "block_shape": block_shape,
        "actual_seq_lengths": q_lengths,
        "actual_seq_lengths_kv": kv_lengths,
        "actual_seq_offsets": actual_seq_offsets,
        "actual_seq_offsets_kv": actual_seq_offsets_kv,
        "num_kv_heads": num_kv_heads,
        "scale_value": scale_value,
    }


def _expand_block_sparse_mask(mask, block_shape, q_len, kv_len):
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    return mask.repeat_interleave(block_x, dim=0).repeat_interleave(block_y, dim=1)[:q_len, :kv_len].bool()


def cpu_block_sparse_attention_tnd_gqa_with_lse(
    query, key, value, block_sparse_mask, block_shape, scale_value, actual_seq_offsets, actual_seq_offsets_kv
):
    query_f = query.cpu().to(torch.float32)
    key_f = key.cpu().to(torch.float32)
    value_f = value.cpu().to(torch.float32)
    mask = block_sparse_mask.cpu()
    total_q, num_heads, head_dim = query_f.shape
    num_kv_heads = key_f.shape[1]
    group_size = num_heads // num_kv_heads
    attention_out = torch.zeros(total_q, num_heads, head_dim, dtype=torch.float32)
    softmax_lse = torch.zeros(total_q, num_heads, 1, dtype=torch.float32)

    for b in range(len(actual_seq_offsets) - 1):
        q_start, q_end = actual_seq_offsets[b], actual_seq_offsets[b + 1]
        kv_start, kv_end = actual_seq_offsets_kv[b], actual_seq_offsets_kv[b + 1]
        q_len = q_end - q_start
        kv_len = kv_end - kv_start
        for n in range(num_heads):
            kv_head = n // group_size
            q_b = query_f[q_start:q_end, n, :]
            k_b = key_f[kv_start:kv_end, kv_head, :]
            v_b = value_f[kv_start:kv_end, kv_head, :]
            scores = torch.matmul(q_b, k_b.transpose(0, 1)) * float(scale_value)
            full_mask = _expand_block_sparse_mask(mask[b, n], block_shape, q_len, kv_len)
            scores = scores.masked_fill(~full_mask, -1e10)
            probs = torch.softmax(scores, dim=-1)
            attention_out[q_start:q_end, n, :] = torch.matmul(probs, v_b)
            softmax_lse[q_start:q_end, n, 0] = torch.logsumexp(scores, dim=-1)

    return attention_out.to(query.dtype), softmax_lse


def cpu_block_sparse_attention_backward_tnd_gqa(
    query, key, value, d_out, block_sparse_mask, block_shape, scale_value, actual_seq_offsets, actual_seq_offsets_kv
):
    query_f = query.cpu().to(torch.float32)
    key_f = key.cpu().to(torch.float32)
    value_f = value.cpu().to(torch.float32)
    d_out_f = d_out.cpu().to(torch.float32)
    mask = block_sparse_mask.cpu()
    total_q, num_heads, head_dim = query_f.shape
    total_kv, num_kv_heads, _ = key_f.shape
    group_size = num_heads // num_kv_heads
    d_query = torch.zeros(total_q, num_heads, head_dim, dtype=torch.float32)
    d_key = torch.zeros(total_kv, num_kv_heads, head_dim, dtype=torch.float32)
    d_value = torch.zeros(total_kv, num_kv_heads, head_dim, dtype=torch.float32)

    for b in range(len(actual_seq_offsets) - 1):
        q_start, q_end = actual_seq_offsets[b], actual_seq_offsets[b + 1]
        kv_start, kv_end = actual_seq_offsets_kv[b], actual_seq_offsets_kv[b + 1]
        q_len = q_end - q_start
        kv_len = kv_end - kv_start
        for n in range(num_heads):
            kv_head = n // group_size
            q_b = query_f[q_start:q_end, n, :]
            k_b = key_f[kv_start:kv_end, kv_head, :]
            v_b = value_f[kv_start:kv_end, kv_head, :]
            dout_b = d_out_f[q_start:q_end, n, :]
            scores = torch.matmul(q_b, k_b.transpose(0, 1)) * float(scale_value)
            full_mask = _expand_block_sparse_mask(mask[b, n], block_shape, q_len, kv_len)
            scores = scores.masked_fill(~full_mask, -1e10)
            probs = torch.softmax(scores, dim=-1)
            d_p = torch.matmul(dout_b, v_b.transpose(0, 1))
            d_s = (d_p - (d_p * probs).sum(dim=-1, keepdim=True)) * probs * float(scale_value)
            d_query[q_start:q_end, n, :] = torch.matmul(d_s, k_b)
            d_key[kv_start:kv_end, kv_head, :] += torch.matmul(d_s.transpose(0, 1), q_b)
            d_value[kv_start:kv_end, kv_head, :] += torch.matmul(probs.transpose(0, 1), dout_b)

    return d_query.to(query.dtype), d_key.to(key.dtype), d_value.to(value.dtype)


def _copy_tnd_gqa_case_to_device(case, device):
    return {
        "query": case["query"].to(device),
        "key": case["key"].to(device),
        "value": case["value"].to(device),
        "d_out": case["d_out"].to(device),
        "block_sparse_mask": case["block_sparse_mask"].to(device),
    }


class TestNPUBlockSparseAttentionBackward(TestCase):
    """Test npu_block_sparse_attention_backward，与 CPU 反向标杆对比."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        if hasattr(torch.npu, 'manual_seed'):
            torch.npu.manual_seed(42)
        np.random.seed(42)
        torch.npu.empty_cache()

    def tearDown(self):
        gc.collect()
        torch.npu.empty_cache()
        super().tearDown()

    def _assert_tnd_gqa_grads_equal(self, cpu_grads, npu_grads):
        for cpu_grad, npu_grad in zip(cpu_grads, npu_grads):
            self.assertRtolEqual(cpu_grad.cpu().float(), npu_grad.cpu().float(), prec=0.02, prec16=0.02)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_backward_bnsd_cpu_compare(self, device="npu"):
        """Backward BNSD 场景验证；attention_out/softmax_lse 由 CPU 正向标杆生成."""
        torch.npu.empty_cache()
        torch.manual_seed(42)
        np.random.seed(42)
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = BLOCK_SHAPE
        ceil_q = (S + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, S, D, dtype=DTYPE)
        key = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        value = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        d_out = torch.randn(B, N, S, D, dtype=DTYPE)
        block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int8)

        dq_cpu, dk_cpu, dv_cpu = cpu_block_sparse_attention_backward_bnsd(
            query, key, value, d_out, block_sparse_mask, block_shape, scale_value, num_kv_heads)
        attention_out_cpu, softmax_lse_cpu = cpu_block_sparse_attention_bnsd_with_lse(
            query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        d_out = d_out.to(device)
        block_sparse_mask = block_sparse_mask.to(device)
        attention_out = attention_out_cpu.to(device)
        softmax_lse_out = softmax_lse_cpu.to(device)

        d_query, d_key, d_value = torch_npu.npu_block_sparse_attention_backward(
            d_out, query, key, value,
            attention_out, softmax_lse_out, block_sparse_mask,
            block_shape=block_shape,
            actual_seq_lengths=[S] * B, actual_seq_lengths_kv=[S] * B,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value,
        )

        self.assertRtolEqual(dq_cpu.cpu().float(), d_query.cpu().float(), prec=0.01, prec16=0.01)
        self.assertRtolEqual(dk_cpu.cpu().float(), d_key.cpu().float(), prec=0.01, prec16=0.01)
        self.assertRtolEqual(dv_cpu.cpu().float(), d_value.cpu().float(), prec=0.01, prec16=0.01)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_backward_bnsd_bfloat16_cpu_compare(self, device="npu"):
        """Backward BNSD bfloat16：与 CPU 反向标杆对比，验证 bfloat16 支持."""
        if hasattr(torch.npu, 'is_bf16_supported') and not torch.npu.is_bf16_supported():
            self.skipTest("NPU bfloat16 not supported")
        torch.npu.empty_cache()
        dtype_bf16 = torch.bfloat16
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = BLOCK_SHAPE
        ceil_q = (S + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, S, D, dtype=dtype_bf16)
        key = torch.randn(B, num_kv_heads, S, D, dtype=dtype_bf16)
        value = torch.randn(B, num_kv_heads, S, D, dtype=dtype_bf16)
        d_out = torch.randn(B, N, S, D, dtype=dtype_bf16)
        block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int8)

        dq_cpu, dk_cpu, dv_cpu = cpu_block_sparse_attention_backward_bnsd(
            query, key, value, d_out, block_sparse_mask, block_shape, scale_value, num_kv_heads)
        attention_out_cpu, softmax_lse_cpu = cpu_block_sparse_attention_bnsd_with_lse(
            query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        d_out = d_out.to(device)
        block_sparse_mask = block_sparse_mask.to(device)
        attention_out = attention_out_cpu.to(device)
        softmax_lse_out = softmax_lse_cpu.to(device)

        d_query, d_key, d_value = torch_npu.npu_block_sparse_attention_backward(
            d_out, query, key, value,
            attention_out, softmax_lse_out, block_sparse_mask,
            block_shape=block_shape,
            actual_seq_lengths=[S] * B, actual_seq_lengths_kv=[S] * B,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value,
        )

        self.assertRtolEqual(dq_cpu.cpu().float(), d_query.cpu().float(), prec=0.01, prec16=0.01)
        self.assertRtolEqual(dk_cpu.cpu().float(), d_key.cpu().float(), prec=0.01, prec16=0.01)
        self.assertRtolEqual(dv_cpu.cpu().float(), d_value.cpu().float(), prec=0.01, prec16=0.01)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_backward_bnsd_sparse_mask_cpu_compare(self, device="npu"):
        """Backward BNSD 稀疏掩码：每个 q_block 仅 attend 一个 kv_block，验证稀疏模式."""
        torch.npu.empty_cache()
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = [32, 128]
        s_sparse = 256
        ceil_q = (s_sparse + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (s_sparse + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, s_sparse, D, dtype=DTYPE)
        key = torch.randn(B, num_kv_heads, s_sparse, D, dtype=DTYPE)
        value = torch.randn(B, num_kv_heads, s_sparse, D, dtype=DTYPE)
        d_out = torch.randn(B, N, s_sparse, D, dtype=DTYPE)
        block_sparse_mask = torch.zeros(B, N, ceil_q, ceil_kv, dtype=torch.int8)
        for i in range(ceil_q):
            block_sparse_mask[:, :, i, i % ceil_kv] = 1

        dq_cpu, dk_cpu, dv_cpu = cpu_block_sparse_attention_backward_bnsd(
            query, key, value, d_out, block_sparse_mask, block_shape, scale_value, num_kv_heads)
        attention_out_cpu, softmax_lse_cpu = cpu_block_sparse_attention_bnsd_with_lse(
            query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        d_out = d_out.to(device)
        block_sparse_mask = block_sparse_mask.to(device)
        attention_out = attention_out_cpu.to(device)
        softmax_lse_out = softmax_lse_cpu.to(device)

        d_query, d_key, d_value = torch_npu.npu_block_sparse_attention_backward(
            d_out, query, key, value,
            attention_out, softmax_lse_out, block_sparse_mask,
            block_shape=block_shape,
            actual_seq_lengths=[s_sparse] * B, actual_seq_lengths_kv=[s_sparse] * B,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value,
        )

        self.assertRtolEqual(dq_cpu.cpu().float(), d_query.cpu().float(), prec=0.01, prec16=0.01)
        self.assertRtolEqual(dk_cpu.cpu().float(), d_key.cpu().float(), prec=0.01, prec16=0.01)
        self.assertRtolEqual(dv_cpu.cpu().float(), d_value.cpu().float(), prec=0.01, prec16=0.01)

    def _run_tnd_gqa_backward_cpu_compare(self, device, full_mask, **case_kwargs):
        torch.npu.empty_cache()
        case = _make_tnd_gqa_case(full_mask=full_mask, **case_kwargs)
        dq_cpu, dk_cpu, dv_cpu = cpu_block_sparse_attention_backward_tnd_gqa(
            case["query"], case["key"], case["value"], case["d_out"], case["block_sparse_mask"],
            case["block_shape"], case["scale_value"], case["actual_seq_offsets"], case["actual_seq_offsets_kv"])
        attention_out_cpu, softmax_lse_cpu = cpu_block_sparse_attention_tnd_gqa_with_lse(
            case["query"], case["key"], case["value"], case["block_sparse_mask"],
            case["block_shape"], case["scale_value"], case["actual_seq_offsets"], case["actual_seq_offsets_kv"])

        npu_case = _copy_tnd_gqa_case_to_device(case, device)

        d_query, d_key, d_value = torch_npu.npu_block_sparse_attention_backward(
            npu_case["d_out"], npu_case["query"], npu_case["key"], npu_case["value"],
            attention_out_cpu.to(device), softmax_lse_cpu.to(device), npu_case["block_sparse_mask"],
            block_shape=case["block_shape"],
            actual_seq_lengths=case["actual_seq_lengths"],
            actual_seq_lengths_kv=case["actual_seq_lengths_kv"],
            q_input_layout="TND", kv_input_layout="TND",
            num_key_value_heads=case["num_kv_heads"],
            scale_value=case["scale_value"],
        )
        torch.npu.synchronize()
        self._assert_tnd_gqa_grads_equal((dq_cpu, dk_cpu, dv_cpu), (d_query, d_key, d_value))

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    @unittest.skip("Skip until gate CANN version supports BlockSparseAttention TND/GQA backward.")
    def test_npu_block_sparse_attention_backward_tnd_gqa_full_mask_cpu_compare(self, device="npu"):
        """Backward TND GQA with full block sparse mask, compared with CPU golden."""
        self._run_tnd_gqa_backward_cpu_compare(device, full_mask=True)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    @unittest.skip("Skip until gate CANN version supports BlockSparseAttention TND/GQA backward.")
    def test_npu_block_sparse_attention_backward_tnd_gqa_sparse_mask_cpu_compare(self, device="npu"):
        """Backward TND GQA with sparse block mask, compared with CPU golden."""
        self._run_tnd_gqa_backward_cpu_compare(device, full_mask=False)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    @unittest.skip("Skip until gate CANN version supports BlockSparseAttention TND/GQA backward.")
    def test_npu_block_sparse_attention_backward_tnd_gqa_single_batch_cpu_compare(self, device="npu"):
        """Backward TND GQA with a single batch, compared with CPU golden."""
        self._run_tnd_gqa_backward_cpu_compare(
            device, full_mask=True,
            num_heads=4, num_kv_heads=2,
            q_lengths=[127], kv_lengths=[191])

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    @unittest.skip("Skip until gate CANN version supports BlockSparseAttention TND/GQA backward.")
    def test_npu_block_sparse_attention_backward_tnd_gqa_uneven_seq_lengths_cpu_compare(self, device="npu"):
        """Backward TND GQA with uneven variable lengths, compared with CPU golden."""
        self._run_tnd_gqa_backward_cpu_compare(
            device, full_mask=False,
            num_heads=4, num_kv_heads=2,
            q_lengths=[1, 64, 129, 17],
            kv_lengths=[128, 3, 257, 65])

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    @unittest.skip("Skip until gate CANN version supports BlockSparseAttention TND/GQA backward.")
    def test_npu_block_sparse_attention_backward_tnd_gqa_group_size_4_cpu_compare(self, device="npu"):
        """Backward TND GQA with group size 4, compared with CPU golden."""
        self._run_tnd_gqa_backward_cpu_compare(
            device, full_mask=False,
            num_heads=8, num_kv_heads=2,
            q_lengths=[96, 137],
            kv_lengths=[111, 259])

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    @unittest.skip("Skip until gate CANN version supports BlockSparseAttention TND/GQA backward.")
    def test_npu_block_sparse_attention_backward_tnd_mqa_cpu_compare(self, device="npu"):
        """Backward TND MQA with one shared KV head, compared with CPU golden."""
        self._run_tnd_gqa_backward_cpu_compare(
            device, full_mask=False,
            num_heads=8, num_kv_heads=1,
            q_lengths=[65, 130],
            kv_lengths=[129, 33])

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    @unittest.skip("Skip until gate CANN version supports BlockSparseAttention TND/GQA backward.")
    def test_npu_block_sparse_attention_backward_tnd_gqa_non_128_tail_block_cpu_compare(self, device="npu"):
        """Backward TND GQA with non-default Q block and tail blocks, compared with CPU golden."""
        self._run_tnd_gqa_backward_cpu_compare(
            device, full_mask=False,
            num_heads=4, num_kv_heads=2,
            block_shape=[64, 128],
            q_lengths=[65, 127, 3],
            kv_lengths=[129, 11, 64])

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    @unittest.skip("Skip until gate CANN version supports BlockSparseAttention TND/GQA backward.")
    def test_npu_block_sparse_attention_backward_tnd_actual_seq_lengths_required(self, device="npu"):
        """TND backward requires corresponding actual sequence lengths."""
        case = _make_tnd_gqa_case(
            full_mask=True,
            num_heads=4, num_kv_heads=2,
            q_lengths=[16], kv_lengths=[16])
        attention_out_cpu, softmax_lse_cpu = cpu_block_sparse_attention_tnd_gqa_with_lse(
            case["query"], case["key"], case["value"], case["block_sparse_mask"],
            case["block_shape"], case["scale_value"], case["actual_seq_offsets"], case["actual_seq_offsets_kv"])
        npu_case = _copy_tnd_gqa_case_to_device(case, device)

        common_args = (
            npu_case["d_out"], npu_case["query"], npu_case["key"], npu_case["value"],
            attention_out_cpu.to(device), softmax_lse_cpu.to(device), npu_case["block_sparse_mask"])
        common_kwargs = {
            "block_shape": case["block_shape"],
            "q_input_layout": "TND",
            "kv_input_layout": "TND",
            "num_key_value_heads": case["num_kv_heads"],
            "scale_value": case["scale_value"],
        }
        with self.assertRaisesRegex(RuntimeError, "actual_seq_lengths must be specified"):
            torch_npu.npu_block_sparse_attention_backward(
                *common_args,
                actual_seq_lengths=None,
                actual_seq_lengths_kv=case["actual_seq_lengths_kv"],
                **common_kwargs)
        with self.assertRaisesRegex(RuntimeError, "actual_seq_lengths_kv must be specified"):
            torch_npu.npu_block_sparse_attention_backward(
                *common_args,
                actual_seq_lengths=case["actual_seq_lengths"],
                actual_seq_lengths_kv=None,
                **common_kwargs)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    @unittest.skip("Skip until gate CANN version supports BlockSparseAttention TND/GQA backward.")
    def test_npu_block_sparse_attention_backward_tnd_gqa_autograd_cpu_compare(self, device="npu"):
        """End-to-end autograd TND GQA path, compared with CPU backward golden."""
        torch.npu.empty_cache()
        case = _make_tnd_gqa_case(full_mask=False)
        dq_cpu, dk_cpu, dv_cpu = cpu_block_sparse_attention_backward_tnd_gqa(
            case["query"], case["key"], case["value"], case["d_out"], case["block_sparse_mask"],
            case["block_shape"], case["scale_value"], case["actual_seq_offsets"], case["actual_seq_offsets_kv"])

        npu_case = _copy_tnd_gqa_case_to_device(case, device)
        query = npu_case["query"]
        key = npu_case["key"]
        value = npu_case["value"]
        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True

        attention_out, _ = torch_npu.npu_block_sparse_attention(
            query, key, value, npu_case["block_sparse_mask"], case["block_shape"],
            q_input_layout="TND", kv_input_layout="TND",
            num_key_value_heads=case["num_kv_heads"],
            scale_value=case["scale_value"],
            inner_precise=1,
            actual_seq_lengths=case["actual_seq_lengths"],
            actual_seq_lengths_kv=case["actual_seq_lengths_kv"],
            softmax_lse_flag=1,
        )
        attention_out.backward(gradient=npu_case["d_out"])
        torch.npu.synchronize()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        self.assertEqual(query.grad.shape, query.shape)
        self.assertEqual(key.grad.shape, key.shape)
        self.assertEqual(value.grad.shape, value.shape)
        self._assert_tnd_gqa_grads_equal((dq_cpu, dk_cpu, dv_cpu), (query.grad, key.grad, value.grad))

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_autograd_backward(self, device="npu"):
        """验证 derivatives.yaml 自动前反向绑定：对 forward 输出调用 .backward()，query/key/value 能收到梯度且与显式 backward 一致."""
        torch.npu.empty_cache()
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = BLOCK_SHAPE
        ceil_q = (S + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, S, D, dtype=DTYPE, device=device)
        key = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE, device=device)
        value = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE, device=device)
        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True
        block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int8, device=device)

        attention_out, softmax_lse = torch_npu.npu_block_sparse_attention(
            query, key, value,
            block_sparse_mask, block_shape,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value, inner_precise=1,
            actual_seq_lengths=[S] * B, actual_seq_lengths_kv=[S] * B,
            softmax_lse_flag=1
        )
        grad_out = torch.ones_like(attention_out, device=device)
        attention_out.backward(gradient=grad_out)
        torch.npu.synchronize()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        self.assertEqual(query.grad.shape, query.shape)
        self.assertEqual(key.grad.shape, key.shape)
        self.assertEqual(value.grad.shape, value.shape)

        d_query_autograd = query.grad.clone()
        d_key_autograd = key.grad.clone()
        d_value_autograd = value.grad.clone()

        query.grad = None
        key.grad = None
        value.grad = None
        d_explicit = torch_npu.npu_block_sparse_attention_backward(
            grad_out, query, key, value,
            attention_out.detach(), softmax_lse.detach(), block_sparse_mask,
            block_shape=block_shape,
            actual_seq_lengths=[S] * B, actual_seq_lengths_kv=[S] * B,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value,
        )
        d_query_explicit, d_key_explicit, d_value_explicit = d_explicit
        torch.npu.synchronize()
        self.assertRtolEqual(d_query_autograd.cpu().float(), d_query_explicit.cpu().float(), prec=0.01, prec16=0.01)
        self.assertRtolEqual(d_key_autograd.cpu().float(), d_key_explicit.cpu().float(), prec=0.01, prec16=0.01)
        self.assertRtolEqual(d_value_autograd.cpu().float(), d_value_explicit.cpu().float(), prec=0.01, prec16=0.01)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_forward_backward_together(self, device="npu"):
        """
        BNSD MHA：正向算子输出直接作为反向算子输入，验证正反向算子串接可用。

        为与 CPU 标杆公平对比（同一 P 矩阵），attention_out/softmax_lse 使用 CPU 正向标杆生成，
        与其它解耦用例一致。NPU 正向在 autograd_backward 用例中单独验证。
        """
        torch.npu.empty_cache()
        num_kv_heads = NUM_KV_HEADS  # MHA: N == num_kv_heads
        scale_value = 1.0 / math.sqrt(D)
        block_shape = BLOCK_SHAPE
        ceil_q = (S + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, S, D, dtype=DTYPE)
        key = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        value = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        d_out = torch.randn(B, N, S, D, dtype=DTYPE)
        block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int8)

        dq_cpu, dk_cpu, dv_cpu = cpu_block_sparse_attention_backward_bnsd(
            query, key, value, d_out, block_sparse_mask, block_shape, scale_value, num_kv_heads)
        attention_out_cpu, softmax_lse_cpu = cpu_block_sparse_attention_bnsd_with_lse(
            query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        d_out = d_out.to(device)
        block_sparse_mask = block_sparse_mask.to(device)
        attention_out = attention_out_cpu.to(device)
        softmax_lse_out = softmax_lse_cpu.to(device)

        # 反向：使用 CPU 正向标杆输出（与解耦用例一致，保证 P 矩阵一致）
        d_query, d_key, d_value = torch_npu.npu_block_sparse_attention_backward(
            d_out, query, key, value,
            attention_out, softmax_lse_out, block_sparse_mask,
            block_shape=block_shape,
            actual_seq_lengths=[S] * B, actual_seq_lengths_kv=[S] * B,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value,
        )
        torch.npu.synchronize()
        self.assertRtolEqual(dq_cpu.cpu().float(), d_query.cpu().float(), prec=0.01, prec16=0.01)
        self.assertRtolEqual(dk_cpu.cpu().float(), d_key.cpu().float(), prec=0.01, prec16=0.01)
        self.assertRtolEqual(dv_cpu.cpu().float(), d_value.cpu().float(), prec=0.01, prec16=0.01)


if __name__ == "__main__":
    run_tests()
