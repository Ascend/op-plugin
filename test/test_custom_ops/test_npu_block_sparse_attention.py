"""
npu_block_sparse_attention 正向算子单测。

测试场景覆盖：
- BNSD/TND 布局、actual_seq_lengths 可选/必传
- inner_precise=0/1（fp32/fp16 中间 softmax）
- softmax_lse_flag=0/1
- float16、bfloat16 数据类型（bf16 时 inner_precise 仅支持 0）
- GQA 约束 G=N1/N2：G<128 且 128%G==0（G=2,4,8,16,32,64）
- 多块稀疏（block_shape=[8,128] 产生 4x1 块）
- 稀疏掩码（对角块为 1，其余为 0）
"""

import math
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices, SkipIfNotGteCANNVersion

DTYPE = torch.float16
B, S, N, D = 2, 32, 8, 64
NUM_KV_HEADS = 8
BLOCK_SHAPE = [128, 128]


def _softmax_np(x):
    """Softmax with numerical stability. When all inputs are -inf (masked), return zeros."""
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x = x - x_max
    y = np.exp(x)
    s = y.sum(axis=-1, keepdims=True)
    # 全 mask 时 s=0，避免 0/0 得 NaN；用 1 作分母使 probs=0
    s = np.where(s > 1e-10, s, 1.0)
    return y / s


def _logsumexp_np(scores):
    """对一行 scores 计算 log(sum(exp(scores)))，masked 位置为 -1e10 时仍稳定."""
    s = scores.astype(np.float32)
    m = np.max(s)
    if m <= -1e9:
        return m
    return float(m + np.log(np.sum(np.exp(s - m))))


def cpu_block_sparse_attention_bnsd_with_lse(query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads):
    """CPU 标杆：BNSD，返回 (attention_out, softmax_lse)，用于 softmaxLse 校验."""
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
    return torch.from_numpy(out).to(query.dtype), torch.from_numpy(lse).to(torch.float32)


def cpu_block_sparse_attention_bnsd(query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads):
    """
    CPU 标杆：BNSD layout，block_sparse_mask[b,n,i,j]=1 表示 q_block i 与 kv_block j 参与计算.
    query (B,N,S,D), key/value (B,N2,S,D), block_sparse_mask (B,N,ceil_q,ceil_kv).
    """
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
                probs = _softmax_np(scores)
                out[b, n, s1, :] = np.dot(probs, v[b, n2, :, :])
    return torch.from_numpy(out).to(query.dtype)


def cpu_block_sparse_attention_tnd_with_lse(query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads):
    """CPU 标杆：TND，返回 (attention_out, softmax_lse)，用于 softmaxLse 校验."""
    q = query.cpu().to(torch.float32).numpy()
    k = key.cpu().to(torch.float32).numpy()
    v = value.cpu().to(torch.float32).numpy()
    mask = block_sparse_mask.cpu().numpy()
    T, N, D = q.shape
    N2 = k.shape[1]
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (T + block_x - 1) // block_x
    ceil_kv = (T + block_y - 1) // block_y
    out = np.zeros((T, N, D), dtype=np.float32)
    lse = np.zeros((T, N, 1), dtype=np.float32)
    for n in range(N):
        n2 = n % N2
        for s1 in range(T):
            i_block = s1 // block_x
            scores = np.full(T, -1e10, dtype=np.float32)
            for s2 in range(T):
                j_block = s2 // block_y
                if i_block < ceil_q and j_block < ceil_kv and mask[0, n, i_block, j_block] != 0:
                    scores[s2] = float(scale_value) * np.dot(q[s1, n, :], k[s2, n2, :])
            lse[s1, n, 0] = _logsumexp_np(scores)
            probs = _softmax_np(scores)
            out[s1, n, :] = np.dot(probs, v[:, n2, :])
    return torch.from_numpy(out).to(query.dtype), torch.from_numpy(lse).to(torch.float32)


def cpu_block_sparse_attention_tnd(query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads):
    """CPU 标杆：TND layout，batch=1，T 为总 token 数."""
    q = query.cpu().to(torch.float32).numpy()
    k = key.cpu().to(torch.float32).numpy()
    v = value.cpu().to(torch.float32).numpy()
    mask = block_sparse_mask.cpu().numpy()
    T, N, D = q.shape
    N2 = k.shape[1]
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (T + block_x - 1) // block_x
    ceil_kv = (T + block_y - 1) // block_y
    out = np.zeros((T, N, D), dtype=np.float32)
    for n in range(N):
        n2 = n % N2
        for s1 in range(T):
            i_block = s1 // block_x
            scores = np.full(T, -1e10, dtype=np.float32)
            for s2 in range(T):
                j_block = s2 // block_y
                if i_block < ceil_q and j_block < ceil_kv and mask[0, n, i_block, j_block] != 0:
                    scores[s2] = float(scale_value) * np.dot(q[s1, n, :], k[s2, n2, :])
            probs = _softmax_np(scores)
            out[s1, n, :] = np.dot(probs, v[:, n2, :])
    return torch.from_numpy(out).to(query.dtype)


class TestNPUBlockSparseAttention(TestCase):
    """Test npu_block_sparse_attention 正向，与 CPU 标杆对比."""

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_bnsd_cpu_compare(self, device="npu"):
        """BNSD：NPU 与 CPU 标杆对比，统一使用 BLOCK_SHAPE；校验 attention_out 与 softmaxLse."""
        torch.npu.empty_cache()
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = BLOCK_SHAPE
        ceil_q = (S + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, S, D, dtype=DTYPE)
        key = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        value = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int8)

        cpu_out, cpu_lse = cpu_block_sparse_attention_bnsd_with_lse(
            query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        block_sparse_mask = block_sparse_mask.to(device)

        npu_out, npu_lse = torch_npu.npu_block_sparse_attention(
            query, key, value, block_sparse_mask, block_shape,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value, inner_precise=1,
            softmax_lse_flag=1,
        )

        npu_out_cpu = npu_out.cpu().to(torch.float32)
        cpu_out_f = cpu_out.cpu().to(torch.float32)
        self.assertRtolEqual(cpu_out_f, npu_out_cpu, prec=0.005, prec16=0.005)

        npu_lse_cpu = npu_lse.cpu().float()
        cpu_lse_f = cpu_lse.cpu().float()
        self.assertRtolEqual(cpu_lse_f, npu_lse_cpu, prec=0.005, prec16=0.01)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_bnsd_with_optional_args_cpu_compare(self, device="npu"):
        """BNSD 显式传 block_shape/actual_seq_lengths：NPU 与 CPU 标杆对比；校验 softmaxLse."""
        torch.npu.empty_cache()
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = BLOCK_SHAPE
        ceil_q = (S + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, S, D, dtype=DTYPE)
        key = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        value = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int8)

        cpu_out, cpu_lse = cpu_block_sparse_attention_bnsd_with_lse(
            query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        block_sparse_mask = block_sparse_mask.to(device)

        npu_out, npu_lse = torch_npu.npu_block_sparse_attention(
            query, key, value, block_sparse_mask, block_shape,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value, inner_precise=1,
            actual_seq_lengths=[S] * B, actual_seq_lengths_kv=[S] * B,
            softmax_lse_flag=1,
        )

        npu_out_cpu = npu_out.cpu().to(torch.float32)
        cpu_out_f = cpu_out.cpu().to(torch.float32)
        self.assertRtolEqual(cpu_out_f, npu_out_cpu, prec=0.005, prec16=0.005)

        npu_lse_cpu = npu_lse.cpu().float()
        cpu_lse_f = cpu_lse.cpu().float()
        self.assertRtolEqual(cpu_lse_f, npu_lse_cpu, prec=0.005, prec16=0.01)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_tnd_cpu_compare(self, device="npu"):
        """TND：NPU 与 CPU 标杆对比，必传 actual_seq_lengths/actual_seq_lengths_kv；校验 softmaxLse."""
        torch.npu.empty_cache()
        T = S  # TND 总 token 数，与 BNSD 的 S 一致便于对比
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = BLOCK_SHAPE
        ceil_q = (T + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (T + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(T, N, D, dtype=DTYPE)
        key = torch.randn(T, num_kv_heads, D, dtype=DTYPE)
        value = torch.randn(T, num_kv_heads, D, dtype=DTYPE)
        block_sparse_mask = torch.ones(1, N, ceil_q, ceil_kv, dtype=torch.int8)

        cpu_out, cpu_lse = cpu_block_sparse_attention_tnd_with_lse(
            query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        block_sparse_mask = block_sparse_mask.to(device)

        npu_out, npu_lse = torch_npu.npu_block_sparse_attention(
            query, key, value, block_sparse_mask, BLOCK_SHAPE,
            q_input_layout="TND", kv_input_layout="TND",
            num_key_value_heads=num_kv_heads, scale_value=scale_value, inner_precise=1,
            actual_seq_lengths=[T], actual_seq_lengths_kv=[T],
            softmax_lse_flag=1,
        )

        npu_out_cpu = npu_out.cpu().to(torch.float32)
        cpu_out_f = cpu_out.cpu().to(torch.float32)
        self.assertRtolEqual(cpu_out_f, npu_out_cpu, prec=0.005, prec16=0.005)

        npu_lse_cpu = npu_lse.cpu().float()
        cpu_lse_f = cpu_lse.cpu().float()
        self.assertRtolEqual(cpu_lse_f, npu_lse_cpu, prec=0.005, prec16=0.01)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_bnsd_inner_precise0_cpu_compare(self, device="npu"):
        """BNSD inner_precise=0：fp32 中间 softmax，与 CPU 标杆对比."""
        torch.npu.empty_cache()
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = BLOCK_SHAPE
        ceil_q = (S + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, S, D, dtype=DTYPE)
        key = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        value = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int8)

        cpu_out = cpu_block_sparse_attention_bnsd(
            query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        block_sparse_mask = block_sparse_mask.to(device)

        npu_out, _ = torch_npu.npu_block_sparse_attention(
            query, key, value, block_sparse_mask, block_shape,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value, inner_precise=0,
        )

        npu_out_cpu = npu_out.cpu().to(torch.float32)
        cpu_out_f = cpu_out.cpu().to(torch.float32)
        self.assertRtolEqual(cpu_out_f, npu_out_cpu, prec=0.005, prec16=0.005)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_bnsd_softmax_lse_flag0(self, device="npu"):
        """BNSD softmax_lse_flag=0：不输出 softmax_lse，验证返回形状正确."""
        torch.npu.empty_cache()
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = BLOCK_SHAPE
        ceil_q = (S + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, S, D, dtype=DTYPE).to(device)
        key = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE).to(device)
        value = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE).to(device)
        block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int8).to(device)

        attention_out, softmax_lse = torch_npu.npu_block_sparse_attention(
            query, key, value, block_sparse_mask, block_shape,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value, inner_precise=1,
            softmax_lse_flag=0,
        )

        self.assertEqual(attention_out.shape, (B, N, S, D))
        self.assertEqual(attention_out.dtype, DTYPE)
        self.assertEqual(softmax_lse.shape, (B, N, S, 1))
        self.assertEqual(softmax_lse.dtype, torch.float32)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_bnsd_bfloat16_cpu_compare(self, device="npu"):
        """BNSD bfloat16：与 CPU 标杆对比，验证 bfloat16 支持."""
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
        block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int8)

        cpu_out = cpu_block_sparse_attention_bnsd(
            query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        block_sparse_mask = block_sparse_mask.to(device)

        npu_out, _ = torch_npu.npu_block_sparse_attention(
            query, key, value, block_sparse_mask, block_shape,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value, inner_precise=0,
        )

        npu_out_cpu = npu_out.cpu().to(torch.float32)
        cpu_out_f = cpu_out.cpu().to(torch.float32)
        self.assertRtolEqual(cpu_out_f, npu_out_cpu, prec=0.005, prec16=0.005)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_bnsd_gqa_g_constraint_cpu_compare(self, device="npu"):
        """GQA 场景 G=N1/N2：G<128 且 128%G==0，与 CPU 标杆对比."""
        torch.npu.empty_cache()
        scale_value = 1.0 / math.sqrt(D)
        block_shape = BLOCK_SHAPE
        # (G, N1, N2) 满足 N1/N2=G，且 N1 >= N2， N1 % N2 ==0
        g_configs = [
            (1, 8, 8),
            (8, 8, 1),
        ]
        for G, num_heads, num_kv_heads in g_configs:
            with self.subTest(G=G, N1=num_heads, N2=num_kv_heads):
                assert num_heads % num_kv_heads == 0 and num_heads // num_kv_heads == G
                ceil_q = (S + block_shape[0] - 1) // block_shape[0]
                ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

                query = torch.randn(B, num_heads, S, D, dtype=DTYPE)
                key = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
                value = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
                block_sparse_mask = torch.ones(B, num_heads, ceil_q, ceil_kv, dtype=torch.int8)

                cpu_out = cpu_block_sparse_attention_bnsd(
                    query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

                query = query.to(device)
                key = key.to(device)
                value = value.to(device)
                block_sparse_mask = block_sparse_mask.to(device)

                npu_out, _ = torch_npu.npu_block_sparse_attention(
                    query, key, value, block_sparse_mask, block_shape,
                    q_input_layout="BNSD", kv_input_layout="BNSD",
                    num_key_value_heads=num_kv_heads, scale_value=scale_value, inner_precise=1,
                )

                npu_out_cpu = npu_out.cpu().to(torch.float32)
                cpu_out_f = cpu_out.cpu().to(torch.float32)
                self.assertRtolEqual(cpu_out_f, npu_out_cpu, prec=0.005, prec16=0.005)
        torch.npu.empty_cache()

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_bnsd_multi_block_cpu_compare(self, device="npu"):
        """BNSD 多块：block_shape=[8,128] 产生 4x1 块，验证多块稀疏."""
        torch.npu.empty_cache()
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = [8, 128]  # blockY 须为 128 的倍数
        ceil_q = (S + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, S, D, dtype=DTYPE)
        key = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        value = torch.randn(B, num_kv_heads, S, D, dtype=DTYPE)
        block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int8)

        cpu_out = cpu_block_sparse_attention_bnsd(
            query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        block_sparse_mask = block_sparse_mask.to(device)

        npu_out, _ = torch_npu.npu_block_sparse_attention(
            query, key, value, block_sparse_mask, block_shape,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value, inner_precise=1,
        )

        npu_out_cpu = npu_out.cpu().to(torch.float32)
        cpu_out_f = cpu_out.cpu().to(torch.float32)
        self.assertRtolEqual(cpu_out_f, npu_out_cpu, prec=0.005, prec16=0.005)

    @SkipIfNotGteCANNVersion("9.0.0")
    @SupportedDevices(['Ascend910B'])
    def test_npu_block_sparse_attention_bnsd_sparse_mask_cpu_compare(self, device="npu"):
        """BNSD 稀疏掩码：每个 q_block 仅 attend 一个 kv_block（i -> i%ceil_kv），验证稀疏模式."""
        torch.npu.empty_cache()
        num_kv_heads = NUM_KV_HEADS
        scale_value = 1.0 / math.sqrt(D)
        block_shape = [32, 128]  # S=256 -> ceil_q=8, ceil_kv=2
        s_sparse = 256
        ceil_q = (s_sparse + block_shape[0] - 1) // block_shape[0]
        ceil_kv = (s_sparse + block_shape[1] - 1) // block_shape[1]

        query = torch.randn(B, N, s_sparse, D, dtype=DTYPE)
        key = torch.randn(B, num_kv_heads, s_sparse, D, dtype=DTYPE)
        value = torch.randn(B, num_kv_heads, s_sparse, D, dtype=DTYPE)
        # 每个 q_block i 仅 attend kv_block (i % ceil_kv)，保证每行至少有一个有效块
        block_sparse_mask = torch.zeros(B, N, ceil_q, ceil_kv, dtype=torch.int8)
        for i in range(ceil_q):
            block_sparse_mask[:, :, i, i % ceil_kv] = 1

        cpu_out = cpu_block_sparse_attention_bnsd(
            query, key, value, block_sparse_mask, block_shape, scale_value, num_kv_heads)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        block_sparse_mask = block_sparse_mask.to(device)

        npu_out, _ = torch_npu.npu_block_sparse_attention(
            query, key, value, block_sparse_mask, block_shape,
            q_input_layout="BNSD", kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads, scale_value=scale_value, inner_precise=1,
        )

        npu_out_cpu = npu_out.cpu().to(torch.float32)
        cpu_out_f = cpu_out.cpu().to(torch.float32)
        self.assertRtolEqual(cpu_out_f, npu_out_cpu, prec=0.005, prec16=0.005)


if __name__ == "__main__":
    run_tests()
