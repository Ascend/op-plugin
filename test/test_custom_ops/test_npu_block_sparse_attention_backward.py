"""
npu_block_sparse_attention_backward 反向算子单测。

当前反向算子仅支持 BNSD MHA 场景（BNSD 布局，num_heads == num_kv_heads）。
与正向解耦用例：attention_out、softmax_lse 由 CPU 正向标杆生成，仅反向在 NPU 执行并与 CPU 反向标杆对比。

测试场景覆盖：
- BNSD MHA（num_heads == num_kv_heads）
- head_dim=128（算子限制 head_dim <= 128，用例覆盖边界）
- float16、bfloat16 数据类型
- 多块稀疏（block_shape=[8,128]）
- 稀疏掩码（每个 q_block 仅 attend 一个 kv_block）
- 正反向算子同时调用（forward 输出作为 backward 输入）
- autograd 前反向绑定

精度说明：为保障 NPU 与 CPU 标杆公平对比，所有与 CPU 对比的用例均使用 CPU 正向标杆生成
attention_out、softmax_lse，确保双方使用同一 P 矩阵。若使用 NPU 正向输出，NPU 的 fp16 与 CPU 的 fp32
P 存在差异，会导致梯度对比间歇性超阈值。
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
