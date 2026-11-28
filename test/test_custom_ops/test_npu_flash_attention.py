from functools import wraps
import math
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.npu._backends import get_soc_version
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from einops import rearrange


def _skip_if_lt_soc_950(fn):
    @wraps(fn)
    def dep_fn(slf, *args, **kwargs):
        if get_soc_version() < 260:
            raise unittest.SkipTest("Only run on Ascend950 and later.")
        return fn(slf, *args, **kwargs)

    return dep_fn


def get_cu_seqlens(seqlens_list):
    cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int64)
    for i in range(len(seqlens_list) + 1):
        cu[i] = sum(seqlens_list[:i])
    return cu


def broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor, dtype):
    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    B, S, D = kv_shape[0], kv_shape[2], kv_shape[3]
    kv_res = torch.zeros([B, numHeads, S, D]).to(dtype)
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
    return kv_res


def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    return y.div(x_sum), x_max, x_sum


def tforward(q, k, v, scale, drop_mask, keep_prob):
    qk = torch.matmul(q, k.permute(0, 1, 3, 2)).mul(scale)
    softmax_res, x_max, x_sum = tsoftmax(qk)
    if drop_mask == None or len(drop_mask.shape) == 0:
        drop_res = softmax_res
    else:
        drop_res = softmax_res * drop_mask * (1.0 / keep_prob)
    y = torch.matmul(drop_res, v)
    return y, softmax_res, x_max, x_sum


def tsoftmax_grad(dp, softmax_res):
    muls = dp * softmax_res
    muls_r = muls.sum(dim=-1, keepdims=True)
    sub_r = dp - muls_r
    res = sub_r * softmax_res
    return res


# pylint:disable = huawei-too-many-arguments
def tbackward(dx, q, k, v, softmax_res, scale, drop_mask=None, keep_prob=1.0):
    dp = torch.matmul(dx, v.permute(0, 1, 3, 2))
    if drop_mask == None or len(drop_mask.shape) == 0:
        drop_res = softmax_res.permute(0, 1, 3, 2)
        dp_drop = dp
    else:
        drop_res = softmax_res.mul(drop_mask).mul(1.0 / keep_prob).permute(0, 1, 3, 2)
        dp_drop = dp * drop_mask * (1.0 / keep_prob)
    dv = torch.matmul(drop_res, dx)
    softmax_grad_res = (tsoftmax_grad(dp_drop, softmax_res) * scale)
    dq = torch.matmul(softmax_grad_res, k)
    dk = torch.matmul(softmax_grad_res.permute(0, 1, 3, 2), q)
    return dq, dk, dv


def get_drop_mask(q, length, seed=2, gen_p=0.2):
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.manual_seed(seed)
    drop_mask_uint8 = torch_npu._npu_dropout_gen_mask(q.npu(), [length], p=gen_p, seed=seed, offset=0, parallel=True,
                                                      sync=False)
    drop_mask_bit_np = np.unpackbits(drop_mask_uint8.cpu().numpy(), count=length, bitorder='little')
    drop_mask_bit = torch.from_numpy(drop_mask_bit_np)
    drop_mask_bit = drop_mask_bit.detach().clone().to(torch.uint8)
    return drop_mask_bit.cpu()


def get_drop_mask_for_npu(B, N, S1, S2, seed=2, gen_p=0.2, device="npu"):
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.manual_seed(seed)
    shape = [B, N, S1, S2]
    drop_mask_uint8 = torch_npu._npu_dropout_gen_mask(
        torch.randn(1, device=device), shape, p=gen_p, seed=seed, offset=0,
        parallel=True, sync=False)
    return drop_mask_uint8


class TestNPUFlashAttention(TestCase):
    def supported_op_exec(self, query, key, value, scale, drop_mask=None, keep_prob=1.0):
        qk = torch.matmul(query, key.transpose(2, 3)).mul(scale)
        softmax_res = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        if drop_mask == None or len(drop_mask.shape) == 0:
            drop_res = softmax_res
        else:
            drop_res = softmax_res * drop_mask * (1.0 / keep_prob)
        output = torch.matmul(drop_res, value)
        output = output.transpose(1, 2)
        output = output.reshape(output.shape[0], output.shape[1], -1)
        return output

    # pylint:disable = huawei-too-many-arguments
    def supported_op_exec_with_softmax_out_tnd(self, q, k, v, dx, seqlens_list_q, seqlens_list_k, keep_prob=1.0):
        gtype = torch.float32

        # 从输入推导维度
        S1, N1, D = q.shape
        S2, N2, _ = k.shape
        B = len(seqlens_list_q)
        scale = 1 / (D ** 0.5)

        cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
        cu_seqlens_k = get_cu_seqlens(seqlens_list_k)

        # 设置drop_mask
        qk_size = seqlens_list_q * seqlens_list_k
        qk_pointer = get_cu_seqlens(qk_size)
        if keep_prob == 1.0:
            drop_mask = torch.tensor(1)
        else:
            drop_mask = get_drop_mask(q, qk_pointer[-1]*N1, seed=2, gen_p=1-keep_prob)

        # 运算golden
        out_golden = torch.zeros_like(q)
        dq_golden = torch.zeros_like(q)
        dk_golden = torch.zeros_like(k)
        dv_golden = torch.zeros_like(v)
        x_max_tnd = torch.empty(0)
        x_sum_tnd = torch.empty(0)
        for i in range(B):
            if seqlens_list_q[i] != 0 and seqlens_list_k[i] != 0:
                qi = q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]
                ki = k[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
                vi = v[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
                qi = rearrange(qi, 's n d -> 1 n s d')
                ki = rearrange(ki, 's n d -> 1 n s d')
                vi = rearrange(vi, 's n d -> 1 n s d')
                # N不等长适配by cdy
                if not (N1 == N2):
                    ki = broadcastKV_sigle(N1, N2, ki, ki.dtype)
                    vi = broadcastKV_sigle(N1, N2, vi, vi.dtype)

                if drop_mask.numel() > 1:
                    drop_maski = drop_mask[(qk_pointer[i]*N1):(qk_pointer[i+1]*N1)].reshape(N1, seqlens_list_q[i],
                                                                                            seqlens_list_k[i])
                else:
                    drop_maski = drop_mask

                # 正向golden运算
                outi_golden, softmax_resi, x_maxi, x_sumi = tforward(qi.to(gtype), ki.to(gtype), vi.to(gtype), scale,
                                                                     drop_maski, keep_prob)
                out_golden[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = rearrange(outi_golden, '1 n s d -> s n d')

                # 记录max,sum
                x_maxi = x_maxi.broadcast_to(1, N1, seqlens_list_q[i], 8)

                x_maxi_tnd = x_maxi.permute(0, 2, 1, 3)
                x_maxi.contiguous().view(-1)
                x_maxi_tnd = x_maxi_tnd.contiguous().view(-1)

                x_sumi = x_sumi.broadcast_to(1, N1, seqlens_list_q[i], 8)
                x_sumi_tnd = x_sumi.permute(0, 2, 1, 3)
                x_sumi.contiguous().view(-1)
                x_sumi_tnd = x_sumi_tnd.contiguous().view(-1)

                x_max_tnd = torch.cat([x_max_tnd, x_maxi_tnd], dim=0)
                x_sum_tnd = torch.cat([x_sum_tnd, x_sumi_tnd], dim=0)

                dxi = dx[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]
                dxi = rearrange(dxi, 's n d -> 1 n s d')

                # 反向golden运算
                dqi_golden, dki_golden, dvi_golden = tbackward(dxi.to(gtype), qi.to(gtype), ki.to(gtype), vi.to(gtype),
                                                                softmax_resi.to(gtype), scale, drop_maski, keep_prob)

                # N不等长适配by cdy
                if not (N1 == N2):
                    G = int(N1 / N2)
                    s2i = seqlens_list_k[i]
                    dki_golden = torch.sum(dki_golden.reshape(1, N2, G, s2i, D), dim=2, keepdim=True).reshape(1, N2,
                                                                                                              s2i, D)
                    dvi_golden = torch.sum(dvi_golden.reshape(1, N2, G, s2i, D), dim=2, keepdim=True).reshape(1, N2,
                                                                                                              s2i, D)
                # 记录dqkv
                dq_golden[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = rearrange(dqi_golden, '1 n s d -> s n d')
                dk_golden[cu_seqlens_k[i]:cu_seqlens_k[i + 1]] = rearrange(dki_golden, '1 n s d -> s n d')
                dv_golden[cu_seqlens_k[i]:cu_seqlens_k[i + 1]] = rearrange(dvi_golden, '1 n s d -> s n d')

        # pylint:disable=too-many-return-values
        return (out_golden.detach(), x_max_tnd.detach().reshape(S1, N1, 8), x_sum_tnd.detach().reshape(S1, N1, 8),
                dx.detach(), dq_golden.detach(), dk_golden.detach(), dv_golden.detach())

    # pylint:disable = huawei-too-many-arguments
    def custom_op_exec(self, query, key, value, head_num, scale, input_layout, actual_seq_qlen=None,
                       actual_seq_kvlen=None, softmax_layout="", keep_prob=1.0):
        return torch_npu.npu_fusion_attention(
            query, key, value, head_num=head_num, input_layout=input_layout, scale=scale,
            actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen, softmax_layout=softmax_layout,
            keep_prob=keep_prob)

    # pylint:disable = huawei-too-many-arguments
    def custom_op_exec_with_dropout_mask(self, query, key, value, head_num, scale, input_layout,
                                          dropout_mask=None, seed=0, offset=0, actual_seq_qlen=None,
                                          actual_seq_kvlen=None, softmax_layout="", keep_prob=1.0):
        return torch_npu.npu_fusion_attention(
            query, key, value, head_num=head_num, input_layout=input_layout, scale=scale,
            actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen, softmax_layout=softmax_layout,
            keep_prob=keep_prob, dropout_mask=dropout_mask, seed=seed, offset=offset)

    def trans_BNSD2BSH(self, tensor: torch.Tensor):
        tensor = torch.transpose(tensor, 1, 2)
        tensor = torch.reshape(tensor, (tensor.shape[0], tensor.shape[1], -1))
        return tensor

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention(self, device="npu"):
        query = torch.randn(1, 32, 128, 128, dtype=torch.float16)
        key = torch.randn(1, 32, 128, 128, dtype=torch.float16)
        value = torch.randn(1, 32, 128, 128, dtype=torch.float16)

        q_npu = self.trans_BNSD2BSH(query).npu()
        k_npu = self.trans_BNSD2BSH(key).npu()
        v_npu = self.trans_BNSD2BSH(value).npu()
        output = self.supported_op_exec(query.to(torch.float32),
                                        key.to(torch.float32),
                                        value.to(torch.float32),
                                        scale=0.08838).to(torch.float16)
        result = self.custom_op_exec(q_npu, k_npu, v_npu, head_num=32, scale=0.08838, input_layout="BSH")
        attention_score = result[0]
        self.assertRtolEqual(output, attention_score)

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_tnd(self, device="npu"):
        B, N1, N2, D = 3, 8, 2, 128
        scale = 1 / (D ** 0.5)
        seqlens_list_q = np.array([1, 2, 3])
        seqlens_list_k = np.array([3, 4, 5])
        cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
        cu_seqlens_k = get_cu_seqlens(seqlens_list_k)
        S1 = seqlens_list_q.sum()
        S2 = seqlens_list_k.sum()
        pttype = torch.float16
        q = 2 * (torch.rand([S1, N1, D]) - 0.5).to(pttype)
        k = 2 * (torch.rand([S2, N2, D]) - 0.5).to(pttype)
        v = 2 * (torch.rand([S2, N2, D]) - 0.5).to(pttype)
        dy = 2 * (torch.rand([S1, N1, D]) - 0.5).to(pttype)

        attention_score_golden, softmax_max_golden, softmax_sum_golden, dx, dq_golden, dk_golden, dv_golden = (
            self.supported_op_exec_with_softmax_out_tnd(q, k, v, dy, seqlens_list_q, seqlens_list_k))

        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        dx = dx.to(device)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        result = self.custom_op_exec(
            q, k, v, N1,
            scale=scale,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
            actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
            softmax_layout="TND")

        attention_score_npu = result[0]
        softmax_max_npu = result[1]
        softmax_sum_npu = result[2]
        self.assertRtolEqual(attention_score_golden, attention_score_npu)
        self.assertRtolEqual(softmax_max_golden, softmax_max_npu)
        self.assertRtolEqual(softmax_sum_golden, softmax_sum_npu)

        # 反向精度比对
        attention_score_npu.backward(dx)
        dq_npu = q.grad
        dk_npu = k.grad
        dv_npu = v.grad

        self.assertRtolEqual(dq_golden, dq_npu)
        self.assertRtolEqual(dk_golden, dk_npu)
        self.assertRtolEqual(dv_golden, dv_npu)

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_with_dropmask(self, device="npu"):
        query = torch.randn(1, 32, 256, 128, dtype=torch.float16)
        key = torch.randn(1, 32, 256, 128, dtype=torch.float16)
        value = torch.randn(1, 32, 256, 128, dtype=torch.float16)
        keep_prob = 0.9
        drop_mask = get_drop_mask(query, 1*32*256*256, seed=2, gen_p=1-keep_prob).reshape(1, 32, 256, 256)

        q_npu = self.trans_BNSD2BSH(query).npu()
        k_npu = self.trans_BNSD2BSH(key).npu()
        v_npu = self.trans_BNSD2BSH(value).npu()
        output = self.supported_op_exec(query.to(torch.float32),
                                        key.to(torch.float32),
                                        value.to(torch.float32),
                                        scale=0.08838,
                                        drop_mask=drop_mask,
                                        keep_prob=keep_prob).to(torch.float16)
        result = self.custom_op_exec(q_npu, k_npu, v_npu, head_num=32, scale=0.08838, input_layout="BSH",
                                     keep_prob=keep_prob)
        attention_score = result[0]
        self.assertRtolEqual(output, attention_score)

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_with_external_dropout_mask(self, device="npu"):
        B, N, S, D = 1, 32, 128, 128
        query = torch.randn(B, N, S, D, dtype=torch.float16)
        key = torch.randn(B, N, S, D, dtype=torch.float16)
        value = torch.randn(B, N, S, D, dtype=torch.float16)
        keep_prob = 0.8
        seed = 123
        offset = 0

        q_npu = self.trans_BNSD2BSH(query).npu()
        k_npu = self.trans_BNSD2BSH(key).npu()
        v_npu = self.trans_BNSD2BSH(value).npu()

        dropout_mask = get_drop_mask_for_npu(B, N, S, S, seed=seed, gen_p=1-keep_prob, device=device)

        attention_score, softmax_max, softmax_sum, softmax_out, out_seed, out_offset, numels = \
            self.custom_op_exec_with_dropout_mask(
                q_npu, k_npu, v_npu, head_num=N, scale=0.08838, input_layout="BSH",
                dropout_mask=dropout_mask, seed=seed, offset=offset, keep_prob=keep_prob)

        self.assertEqual(attention_score.shape, q_npu.shape)
        self.assertEqual(softmax_max.shape[0], B)
        self.assertEqual(softmax_max.shape[1], N)
        self.assertEqual(softmax_sum.shape[0], B)
        self.assertEqual(softmax_sum.shape[1], N)

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_dropout_mask_reproducibility(self, device="npu"):
        B, N, S, D = 1, 32, 128, 128
        query = torch.randn(B, N, S, D, dtype=torch.float16)
        key = torch.randn(B, N, S, D, dtype=torch.float16)
        value = torch.randn(B, N, S, D, dtype=torch.float16)
        keep_prob = 0.9
        seed = 456
        offset = 0

        q_npu = self.trans_BNSD2BSH(query).npu()
        k_npu = self.trans_BNSD2BSH(key).npu()
        v_npu = self.trans_BNSD2BSH(value).npu()

        dropout_mask = get_drop_mask_for_npu(B, N, S, S, seed=seed, gen_p=1-keep_prob, device=device)

        result1, _, _, _, _, _, _ = self.custom_op_exec_with_dropout_mask(
            q_npu, k_npu, v_npu, head_num=N, scale=0.125, input_layout="BSH",
            dropout_mask=dropout_mask, seed=seed, offset=offset, keep_prob=keep_prob)

        result2, _, _, _, _, _, _ = self.custom_op_exec_with_dropout_mask(
            q_npu, k_npu, v_npu, head_num=N, scale=0.125, input_layout="BSH",
            dropout_mask=dropout_mask, seed=seed, offset=offset, keep_prob=keep_prob)

        self.assertRtolEqual(result1, result2)

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_without_dropout_mask(self, device="npu"):
        B, N, S, D = 1, 32, 128, 128
        query = torch.randn(B, N, S, D, dtype=torch.float16)
        key = torch.randn(B, N, S, D, dtype=torch.float16)
        value = torch.randn(B, N, S, D, dtype=torch.float16)
        keep_prob = 1.0
        seed = 0
        offset = 0

        q_npu = self.trans_BNSD2BSH(query).npu()
        k_npu = self.trans_BNSD2BSH(key).npu()
        v_npu = self.trans_BNSD2BSH(value).npu()

        attention_score, softmax_max, softmax_sum, softmax_out, out_seed, out_offset, numels = \
            self.custom_op_exec_with_dropout_mask(
                q_npu, k_npu, v_npu, head_num=N, scale=0.125, input_layout="BSH",
                dropout_mask=None, seed=seed, offset=offset, keep_prob=keep_prob)

        self.assertEqual(attention_score.shape, q_npu.shape)

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_tnd_with_dropmask(self, device="npu"):
        B, N1, N2, D = 3, 8, 2, 128
        scale = 1 / (D ** 0.5)
        seqlens_list_q = np.array([1, 2, 3])
        seqlens_list_k = np.array([3, 4, 5])
        cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
        cu_seqlens_k = get_cu_seqlens(seqlens_list_k)
        S1 = seqlens_list_q.sum()
        S2 = seqlens_list_k.sum()
        pttype = torch.float16
        keep_prob = 0.9
        q = 2 * (torch.rand([S1, N1, D]) - 0.5).to(pttype)
        k = 2 * (torch.rand([S2, N2, D]) - 0.5).to(pttype)
        v = 2 * (torch.rand([S2, N2, D]) - 0.5).to(pttype)
        dy = 2 * (torch.rand([S1, N1, D]) - 0.5).to(pttype)

        attention_score_golden, softmax_max_golden, softmax_sum_golden, dx, dq_golden, dk_golden, dv_golden = (
            self.supported_op_exec_with_softmax_out_tnd(q, k, v, dy, seqlens_list_q, seqlens_list_k, keep_prob))

        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        dx = dx.to(device)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        result = self.custom_op_exec(
            q, k, v, N1,
            scale=scale,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
            actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
            softmax_layout="TND",
            keep_prob=keep_prob)

        attention_score_npu = result[0]
        softmax_max_npu = result[1]
        softmax_sum_npu = result[2]
        self.assertRtolEqual(attention_score_golden, attention_score_npu)
        self.assertRtolEqual(softmax_max_golden, softmax_max_npu)
        self.assertRtolEqual(softmax_sum_golden, softmax_sum_npu)

        # 反向精度比对
        attention_score_npu.backward(dx)
        dq_npu = q.grad
        dk_npu = k.grad
        dv_npu = v.grad

        self.assertRtolEqual(dq_golden, dq_npu)
        self.assertRtolEqual(dk_golden, dk_npu)
        self.assertRtolEqual(dv_golden, dv_npu)

    # pylint:disable = huawei-too-many-arguments
    def custom_op_tensor_exec(self, query, key, value, head_num, scale, input_layout, actual_seq_qlen=None,
                       actual_seq_kvlen=None, softmax_layout="", keep_prob=1.0):
        return torch_npu.npu_fusion_attention_v3(
            query, key, value, head_num=head_num, input_layout=input_layout, scale=scale,
            actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen, softmax_layout=softmax_layout,
            keep_prob=keep_prob)

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_tensor(self, device="npu"):
        query = torch.randn(1, 32, 128, 128, dtype=torch.float16)
        key = torch.randn(1, 32, 128, 128, dtype=torch.float16)
        value = torch.randn(1, 32, 128, 128, dtype=torch.float16)

        q_npu = self.trans_BNSD2BSH(query).npu()
        k_npu = self.trans_BNSD2BSH(key).npu()
        v_npu = self.trans_BNSD2BSH(value).npu()
        output = self.supported_op_exec(query.to(torch.float32),
                                        key.to(torch.float32),
                                        value.to(torch.float32),
                                        scale=0.08838).to(torch.float16)
        result = self.custom_op_tensor_exec(q_npu, k_npu, v_npu, head_num=32, scale=0.08838, input_layout="BSH")
        attention_score = result[0]
        self.assertRtolEqual(output, attention_score)

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_tensor_tnd(self, device="npu"):
        B, N1, N2, D = 3, 8, 2, 128
        scale = 1 / (D ** 0.5)
        seqlens_list_q = np.array([1, 2, 3])
        seqlens_list_k = np.array([3, 4, 5])
        cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
        cu_seqlens_k = get_cu_seqlens(seqlens_list_k)
        S1 = seqlens_list_q.sum()
        S2 = seqlens_list_k.sum()
        pttype = torch.float16
        q = 2 * (torch.rand([S1, N1, D]) - 0.5).to(pttype)
        k = 2 * (torch.rand([S2, N2, D]) - 0.5).to(pttype)
        v = 2 * (torch.rand([S2, N2, D]) - 0.5).to(pttype)
        dy = 2 * (torch.rand([S1, N1, D]) - 0.5).to(pttype)

        attention_score_golden, softmax_max_golden, softmax_sum_golden, dx, dq_golden, dk_golden, dv_golden = (
            self.supported_op_exec_with_softmax_out_tnd(q, k, v, dy, seqlens_list_q, seqlens_list_k))

        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        dx = dx.to(device)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        result = self.custom_op_tensor_exec(
            q, k, v, N1,
            scale=scale,
            input_layout="TND",
            actual_seq_qlen=cu_seqlens_q[1:].cpu(),
            actual_seq_kvlen=cu_seqlens_k[1:].cpu(),
            softmax_layout="TND")

        attention_score_npu = result[0]
        softmax_max_npu = result[1]
        softmax_sum_npu = result[2]
        self.assertRtolEqual(attention_score_golden, attention_score_npu)
        self.assertRtolEqual(softmax_max_golden, softmax_max_npu)
        self.assertRtolEqual(softmax_sum_golden, softmax_sum_npu)

        # 反向精度比对
        attention_score_npu.backward(dx)
        dq_npu = q.grad
        dk_npu = k.grad
        dv_npu = v.grad

        self.assertRtolEqual(dq_golden, dq_npu)
        self.assertRtolEqual(dk_golden, dk_npu)
        self.assertRtolEqual(dv_golden, dv_npu)

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_tensor_with_dropmask(self, device="npu"):
        query = torch.randn(1, 32, 256, 128, dtype=torch.float16)
        key = torch.randn(1, 32, 256, 128, dtype=torch.float16)
        value = torch.randn(1, 32, 256, 128, dtype=torch.float16)
        keep_prob = 0.9
        drop_mask = get_drop_mask(query, 1*32*256*256, seed=2, gen_p=1-keep_prob).reshape(1, 32, 256, 256)

        q_npu = self.trans_BNSD2BSH(query).npu()
        k_npu = self.trans_BNSD2BSH(key).npu()
        v_npu = self.trans_BNSD2BSH(value).npu()
        output = self.supported_op_exec(query.to(torch.float32),
                                        key.to(torch.float32),
                                        value.to(torch.float32),
                                        scale=0.08838,
                                        drop_mask=drop_mask,
                                        keep_prob=keep_prob).to(torch.float16)
        result = self.custom_op_tensor_exec(q_npu, k_npu, v_npu, head_num=32, scale=0.08838, input_layout="BSH",
                                     keep_prob=keep_prob)
        attention_score = result[0]
        self.assertRtolEqual(output, attention_score)

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_tensor_tnd_with_dropmask(self, device="npu"):
        B, N1, N2, D = 3, 8, 2, 128
        scale = 1 / (D ** 0.5)
        seqlens_list_q = np.array([1, 2, 3])
        seqlens_list_k = np.array([3, 4, 5])
        cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
        cu_seqlens_k = get_cu_seqlens(seqlens_list_k)
        S1 = seqlens_list_q.sum()
        S2 = seqlens_list_k.sum()
        pttype = torch.float16
        keep_prob = 0.9
        q = 2 * (torch.rand([S1, N1, D]) - 0.5).to(pttype)
        k = 2 * (torch.rand([S2, N2, D]) - 0.5).to(pttype)
        v = 2 * (torch.rand([S2, N2, D]) - 0.5).to(pttype)
        dy = 2 * (torch.rand([S1, N1, D]) - 0.5).to(pttype)

        attention_score_golden, softmax_max_golden, softmax_sum_golden, dx, dq_golden, dk_golden, dv_golden = (
            self.supported_op_exec_with_softmax_out_tnd(q, k, v, dy, seqlens_list_q, seqlens_list_k, keep_prob))

        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        dx = dx.to(device)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        result = self.custom_op_tensor_exec(
            q, k, v, N1,
            scale=scale,
            input_layout="TND",
            actual_seq_qlen=cu_seqlens_q[1:].cpu(),
            actual_seq_kvlen=cu_seqlens_k[1:].cpu(),
            softmax_layout="TND",
            keep_prob=keep_prob)

        attention_score_npu = result[0]
        softmax_max_npu = result[1]
        softmax_sum_npu = result[2]
        self.assertRtolEqual(attention_score_golden, attention_score_npu)
        self.assertRtolEqual(softmax_max_golden, softmax_max_npu)
        self.assertRtolEqual(softmax_sum_golden, softmax_sum_npu)

        # 反向精度比对
        attention_score_npu.backward(dx)
        dq_npu = q.grad
        dk_npu = k.grad
        dv_npu = v.grad

        self.assertRtolEqual(dq_golden, dq_npu)
        self.assertRtolEqual(dk_golden, dk_npu)
        self.assertRtolEqual(dv_golden, dv_npu)

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_tensor_graph(self):
        seed = 558
        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)
        s = torch.npu.get_rng_state()

        query = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")
        key = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")
        value = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")

        result = self.custom_op_tensor_exec(query, key, value, head_num=32, scale=0.08838, input_layout="BSH")
        result1 = result[0].clone()

        query = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")
        key = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")
        value = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")

        result = self.custom_op_tensor_exec(query, key, value, head_num=32, scale=0.08838, input_layout="BSH")
        result2 = result[0].clone()

        g = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(g):
            query = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")
            key = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")
            value = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")

            result = self.custom_op_tensor_exec(query, key, value, head_num=32, scale=0.08838, input_layout="BSH")

        torch.npu.set_rng_state(s)
        result[0].copy_(torch.zeros_like(result[0], device="npu"))
        result[1].copy_(torch.zeros_like(result[1], device="npu"))
        result[2].copy_(torch.zeros_like(result[2], device="npu"))
        result[3].copy_(torch.zeros_like(result[3], device="npu"))
        g.replay()
        self.assertEqual(result[0], result1)
        g.replay()
        self.assertEqual(result[0], result2)

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_tensor_tnd_graph(self, device="npu"):
        seed = 558
        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)
        s = torch.npu.get_rng_state()

        B, N1, N2, D = 3, 8, 2, 128
        scale = 1 / (D ** 0.5)
        seqlens_list_q = np.array([1, 2, 3])
        seqlens_list_k = np.array([3, 4, 5])
        cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
        cu_seqlens_k = get_cu_seqlens(seqlens_list_k)
        S1 = seqlens_list_q.sum()
        S2 = seqlens_list_k.sum()
        pttype = torch.float16
        q = 2 * (torch.rand([S1, N1, D], device="npu") - 0.5).to(pttype)
        k = 2 * (torch.rand([S2, N2, D], device="npu") - 0.5).to(pttype)
        v = 2 * (torch.rand([S2, N2, D], device="npu") - 0.5).to(pttype)
        dy = 2 * (torch.rand([S1, N1, D], device="npu") - 0.5).to(pttype)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        result = self.custom_op_tensor_exec(
            q, k, v, N1,
            scale=scale,
            input_layout="TND",
            actual_seq_qlen=cu_seqlens_q[1:].cpu(),
            actual_seq_kvlen=cu_seqlens_k[1:].cpu(),
            softmax_layout="TND")

        # 反向精度比对
        result[0].backward(dy)
        result1 = result[0].clone()

        q = 2 * (torch.rand([S1, N1, D], device="npu") - 0.5).to(pttype)
        k = 2 * (torch.rand([S2, N2, D], device="npu") - 0.5).to(pttype)
        v = 2 * (torch.rand([S2, N2, D], device="npu") - 0.5).to(pttype)
        dy = 2 * (torch.rand([S1, N1, D], device="npu") - 0.5).to(pttype)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        result = self.custom_op_tensor_exec(
            q, k, v, N1,
            scale=scale,
            input_layout="TND",
            actual_seq_qlen=cu_seqlens_q[1:].cpu(),
            actual_seq_kvlen=cu_seqlens_k[1:].cpu(),
            softmax_layout="TND")

        # 反向精度比对
        result[0].backward(dy)
        result2 = result[0].clone()

        g = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(g):
            q = 2 * (torch.rand([S1, N1, D], device="npu") - 0.5).to(pttype)
            k = 2 * (torch.rand([S2, N2, D], device="npu") - 0.5).to(pttype)
            v = 2 * (torch.rand([S2, N2, D], device="npu") - 0.5).to(pttype)
            dy = 2 * (torch.rand([S1, N1, D], device="npu") - 0.5).to(pttype)
            q.requires_grad = True
            k.requires_grad = True
            v.requires_grad = True
            result = self.custom_op_tensor_exec(
                q, k, v, N1,
                scale=scale,
                input_layout="TND",
                actual_seq_qlen=cu_seqlens_q[1:].cpu(),
                actual_seq_kvlen=cu_seqlens_k[1:].cpu(),
                softmax_layout="TND")
            # 反向精度比对
            result[0].backward(dy)

        torch.npu.set_rng_state(s)
        result[0].copy_(torch.zeros_like(result[0], device="npu"))
        result[1].copy_(torch.zeros_like(result[1], device="npu"))
        result[2].copy_(torch.zeros_like(result[2], device="npu"))
        result[3].copy_(torch.zeros_like(result[3], device="npu"))
        g.replay()
        self.assertEqual(result[0], result1)
        g.replay()
        self.assertEqual(result[0], result2)

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_tensor_with_dropmask_graph(self, device="npu"):
        seed = 558
        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)
        s = torch.npu.get_rng_state()

        keep_prob = 0.9
        query = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")
        key = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")
        value = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")

        result = self.custom_op_tensor_exec(query, key, value, head_num=32, scale=0.08838, input_layout="BSH",
                                     keep_prob=keep_prob)
        result1 = result[0].clone()

        query = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")
        key = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")
        value = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")

        result = self.custom_op_tensor_exec(query, key, value, head_num=32, scale=0.08838, input_layout="BSH",
                                     keep_prob=keep_prob)
        result2 = result[0].clone()

        g = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(g):
            query = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")
            key = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")
            value = torch.randn(1, 128, 4096, dtype=torch.float16, device="npu")

            result = self.custom_op_tensor_exec(query, key, value, head_num=32, scale=0.08838, input_layout="BSH",
                                         keep_prob=keep_prob)

        torch.npu.set_rng_state(s)
        result[0].copy_(torch.zeros_like(result[0], device="npu"))
        result[1].copy_(torch.zeros_like(result[1], device="npu"))
        result[2].copy_(torch.zeros_like(result[2], device="npu"))
        result[3].copy_(torch.zeros_like(result[3], device="npu"))
        g.replay()
        self.assertEqual(result[0], result1)
        g.replay()
        self.assertEqual(result[0], result2)

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_tensor_tnd_with_dropmask(self, device="npu"):
        seed = 558
        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)
        s = torch.npu.get_rng_state()

        B, N1, N2, D = 3, 8, 2, 128
        scale = 1 / (D ** 0.5)
        seqlens_list_q = np.array([1, 2, 3])
        seqlens_list_k = np.array([3, 4, 5])
        cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
        cu_seqlens_k = get_cu_seqlens(seqlens_list_k)
        S1 = seqlens_list_q.sum()
        S2 = seqlens_list_k.sum()
        pttype = torch.float16
        keep_prob = 0.9
        q = 2 * (torch.rand([S1, N1, D], device="npu") - 0.5).to(pttype)
        k = 2 * (torch.rand([S2, N2, D], device="npu") - 0.5).to(pttype)
        v = 2 * (torch.rand([S2, N2, D], device="npu") - 0.5).to(pttype)
        dy = 2 * (torch.rand([S1, N1, D], device="npu") - 0.5).to(pttype)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        result = self.custom_op_tensor_exec(
            q, k, v, N1,
            scale=scale,
            input_layout="TND",
            actual_seq_qlen=cu_seqlens_q[1:].cpu(),
            actual_seq_kvlen=cu_seqlens_k[1:].cpu(),
            softmax_layout="TND",
            keep_prob=keep_prob)

        # 反向精度比对
        result[0].backward(dy)
        result1 = result[0].clone()

        q = 2 * (torch.rand([S1, N1, D], device="npu") - 0.5).to(pttype)
        k = 2 * (torch.rand([S2, N2, D], device="npu") - 0.5).to(pttype)
        v = 2 * (torch.rand([S2, N2, D], device="npu") - 0.5).to(pttype)
        dy = 2 * (torch.rand([S1, N1, D], device="npu") - 0.5).to(pttype)
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        result = self.custom_op_tensor_exec(
            q, k, v, N1,
            scale=scale,
            input_layout="TND",
            actual_seq_qlen=cu_seqlens_q[1:].cpu(),
            actual_seq_kvlen=cu_seqlens_k[1:].cpu(),
            softmax_layout="TND",
            keep_prob=keep_prob)

        # 反向精度比对
        result[0].backward(dy)
        result2 = result[0].clone()

        g = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(g):
            q = 2 * (torch.rand([S1, N1, D], device="npu") - 0.5).to(pttype)
            k = 2 * (torch.rand([S2, N2, D], device="npu") - 0.5).to(pttype)
            v = 2 * (torch.rand([S2, N2, D], device="npu") - 0.5).to(pttype)
            dy = 2 * (torch.rand([S1, N1, D], device="npu") - 0.5).to(pttype)
            q.requires_grad = True
            k.requires_grad = True
            v.requires_grad = True
            result = self.custom_op_tensor_exec(
                q, k, v, N1,
                scale=scale,
                input_layout="TND",
                actual_seq_qlen=cu_seqlens_q[1:].cpu(),
                actual_seq_kvlen=cu_seqlens_k[1:].cpu(),
                softmax_layout="TND",
                keep_prob=keep_prob)
            # 反向精度比对
            result[0].backward(dy)

        torch.npu.set_rng_state(s)
        result[0].copy_(torch.zeros_like(result[0], device="npu"))
        result[1].copy_(torch.zeros_like(result[1], device="npu"))
        result[2].copy_(torch.zeros_like(result[2], device="npu"))
        result[3].copy_(torch.zeros_like(result[3], device="npu"))
        g.replay()
        self.assertEqual(result[0], result1)
        g.replay()
        self.assertEqual(result[0], result2)

    @_skip_if_lt_soc_950
    def test_npu_flash_attention_v3(self, device="npu"):
        for dtype in (torch.float16, torch.bfloat16):
            query = torch.randn(2, 8, 64, 64, dtype=dtype, device="npu", requires_grad=True)
            key = torch.randn(2, 8, 64, 64, dtype=dtype, device="npu", requires_grad=True)
            value = torch.randn(2, 8, 64, 64, dtype=dtype, device="npu", requires_grad=True)
            scale = 1.0 / math.sqrt(64)

            ref_q = query.detach().float().cpu().requires_grad_(True)
            ref_k = key.detach().float().cpu().requires_grad_(True)
            ref_v = value.detach().float().cpu().requires_grad_(True)
            ref_out = torch.nn.functional.scaled_dot_product_attention(ref_q, ref_k, ref_v, scale=scale)
            ref_out.sum().backward()

            out_v3 = torch_npu.npu_fusion_attention_v3(
                query, key, value, head_num=8, input_layout="BNSD", scale=scale, keep_prob=1.0
            )
            self.assertEqual(tuple(out_v3[4].shape), (1,))
            self.assertEqual(tuple(out_v3[5].shape), (1,))
            self.assertEqual(out_v3[4].item(), 0)
            self.assertEqual(out_v3[5].item(), 0)
            out_v3[0].sum().backward()

            atol = 2e-2 if dtype == torch.bfloat16 else 5e-3
            self.assertTrue(torch.allclose(out_v3[0].float().cpu(), ref_out, rtol=atol, atol=atol))
            self.assertTrue(torch.allclose(query.grad.float().cpu(), ref_q.grad, rtol=atol, atol=atol))
            self.assertTrue(torch.allclose(key.grad.float().cpu(), ref_k.grad, rtol=atol, atol=atol))
            self.assertTrue(torch.allclose(value.grad.float().cpu(), ref_v.grad, rtol=atol, atol=atol))

            # Dropout training path via Tensor seed/offset.
            rng_state = torch.npu.get_rng_state()
            q_d = query.detach().clone().requires_grad_(True)
            k_d = key.detach().clone().requires_grad_(True)
            v_d = value.detach().clone().requires_grad_(True)
            out_drop = torch_npu.npu_fusion_attention_v3(
                q_d, k_d, v_d, head_num=8, input_layout="BNSD", scale=scale, keep_prob=0.9
            )
            self.assertFalse(torch.equal(out_drop[0], out_v3[0]))
            out_drop[0].sum().backward()
            self.assertTrue(torch.isfinite(q_d.grad).all())
            self.assertTrue(torch.isfinite(k_d.grad).all())
            self.assertTrue(torch.isfinite(v_d.grad).all())

            # Determinism under the same RNG state.
            torch.npu.set_rng_state(rng_state)
            q_d2 = query.detach().clone().requires_grad_(True)
            k_d2 = key.detach().clone().requires_grad_(True)
            v_d2 = value.detach().clone().requires_grad_(True)
            out_drop2 = torch_npu.npu_fusion_attention_v3(
                q_d2, k_d2, v_d2, head_num=8, input_layout="BNSD", scale=scale, keep_prob=0.9
            )
            out_drop2[0].sum().backward()
            self.assertTrue(torch.equal(out_drop[4], out_drop2[4]))
            self.assertTrue(torch.equal(out_drop[5], out_drop2[5]))
            self.assertTrue(torch.equal(out_drop[0], out_drop2[0]))
            self.assertTrue(torch.equal(q_d.grad, q_d2.grad))
            self.assertTrue(torch.equal(k_d.grad, k_d2.grad))
            self.assertTrue(torch.equal(v_d.grad, v_d2.grad))

    @SupportedDevices(['Ascend950'])
    def test_npu_flash_attention_v3_scalar_rng_contract_ascend950(self, device="npu"):
        for dtype in (torch.float16, torch.bfloat16):
            query = torch.randn(2, 8, 64, 64, dtype=dtype, device="npu", requires_grad=True)
            key = torch.randn(2, 8, 64, 64, dtype=dtype, device="npu", requires_grad=True)
            value = torch.randn(2, 8, 64, 64, dtype=dtype, device="npu", requires_grad=True)
            scale = 1.0 / math.sqrt(64)

            out_nodrop = torch_npu.npu_fusion_attention_v3(
                query, key, value, head_num=8, input_layout="BNSD", scale=scale, keep_prob=1.0
            )
            self.assertEqual(out_nodrop[4].device.type, "cpu")
            self.assertEqual(out_nodrop[5].device.type, "cpu")

            rng_state = torch.npu.get_rng_state()
            q_v3 = query.detach().clone().requires_grad_(True)
            k_v3 = key.detach().clone().requires_grad_(True)
            v_v3 = value.detach().clone().requires_grad_(True)
            out_v3 = torch_npu.npu_fusion_attention_v3(
                q_v3, k_v3, v_v3, head_num=8, input_layout="BNSD", scale=scale, keep_prob=0.9
            )
            self.assertEqual(out_v3[4].device.type, "cpu")
            self.assertEqual(out_v3[5].device.type, "cpu")
            out_v3[0].sum().backward()

            torch.npu.set_rng_state(rng_state)
            q_v1 = query.detach().clone().requires_grad_(True)
            k_v1 = key.detach().clone().requires_grad_(True)
            v_v1 = value.detach().clone().requires_grad_(True)
            out_v1 = torch_npu.npu_fusion_attention(
                q_v1, k_v1, v_v1, head_num=8, input_layout="BNSD", scale=scale, keep_prob=0.9
            )
            out_v1[0].sum().backward()
            self.assertEqual(out_v3[4].item(), out_v1[4])
            self.assertEqual(out_v3[5].item(), out_v1[5])
            self.assertTrue(torch.equal(out_v3[0], out_v1[0]))
            self.assertTrue(torch.equal(q_v3.grad, q_v1.grad))
            self.assertTrue(torch.equal(k_v3.grad, k_v1.grad))
            self.assertTrue(torch.equal(v_v3.grad, v_v1.grad))

    @_skip_if_lt_soc_950
    def test_sdpa_dropout_backward(self, device="npu"):
        q = torch.randn(2, 8, 64, 64, dtype=torch.bfloat16, device="npu", requires_grad=True)
        k = torch.randn(2, 8, 64, 64, dtype=torch.bfloat16, device="npu", requires_grad=True)
        v = torch.randn(2, 8, 64, 64, dtype=torch.bfloat16, device="npu", requires_grad=True)
        rng_state = torch.npu.get_rng_state()
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.1, is_causal=True)
        out.sum().backward()
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue(torch.isfinite(k.grad).all())
        self.assertTrue(torch.isfinite(v.grad).all())

        torch.npu.set_rng_state(rng_state)
        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        out2 = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, dropout_p=0.1, is_causal=True)
        out2.sum().backward()
        self.assertTrue(torch.equal(out, out2))
        self.assertTrue(torch.equal(q.grad, q2.grad))
        self.assertTrue(torch.equal(k.grad, k2.grad))
        self.assertTrue(torch.equal(v.grad, v2.grad))

        with torch.no_grad():
            out_nodrop = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        self.assertFalse(torch.equal(out.detach(), out_nodrop))

    @SupportedDevices(['Ascend950'])
    def test_npu_flash_attention_v3_capture_dropout_ascend950(self, device="npu"):
        def _run_graph_case(fn, q0, k0, v0, dy, extra_args=(), nodrop_fn=None):
            q = q0.detach().clone().requires_grad_(True)
            k = k0.detach().clone().requires_grad_(True)
            v = v0.detach().clone().requires_grad_(True)
            s = torch_npu.npu.Stream()
            s.wait_stream(torch_npu.npu.current_stream())
            with torch_npu.npu.stream(s):
                fn(q, k, v, *extra_args).backward(dy)
            torch_npu.npu.current_stream().wait_stream(s)
            torch_npu.npu.synchronize()
            q.grad.zero_()
            k.grad.zero_()
            v.grad.zero_()

            g = torch_npu.npu.NPUGraph()
            with torch_npu.npu.graph(g):
                out = fn(q, k, v, *extra_args)
                out.backward(dy)
            torch_npu.npu.synchronize()

            # Determinism under same seed
            torch.npu.manual_seed(777)
            q.grad.zero_()
            k.grad.zero_()
            v.grad.zero_()
            g.replay()
            torch_npu.npu.synchronize()
            o1, dq1, dk1, dv1 = out.detach().clone(), q.grad.clone(), k.grad.clone(), v.grad.clone()

            # Adjoint linearity check: <dy, out> == <dv, v> proves fwd & bwd share identical drop_mask
            lhs = float((o1.float() * dy.float()).sum().item())
            rhs = float((v.detach().float() * dv1.float()).sum().item())
            self.assertLess(abs(lhs - rhs) / max(abs(lhs), 1e-6), 0.02)

            torch.npu.manual_seed(777)
            q.grad.zero_()
            k.grad.zero_()
            v.grad.zero_()
            g.replay()
            torch_npu.npu.synchronize()
            self.assertTrue(torch.equal(o1, out.detach()))
            self.assertTrue(torch.equal(dq1, q.grad))
            self.assertTrue(torch.equal(dk1, k.grad))
            self.assertTrue(torch.equal(dv1, v.grad))

            # Consecutive replay without reseed must advance RNG
            q.grad.zero_()
            k.grad.zero_()
            v.grad.zero_()
            g.replay()
            torch_npu.npu.synchronize()
            self.assertFalse(torch.equal(o1, out.detach()))
            self.assertFalse(torch.equal(dq1, q.grad))

            if nodrop_fn is not None:
                with torch.no_grad():
                    o_nodrop = nodrop_fn(q0, k0, v0, *extra_args)
                self.assertFalse(torch.equal(o1, o_nodrop))

        scale = 1.0 / math.sqrt(64)
        # 1. FAV3 BNSD + 2. SDPA BNSD
        q_b = torch.randn(2, 8, 64, 64, dtype=torch.bfloat16, device="npu")
        k_b = torch.randn(2, 8, 64, 64, dtype=torch.bfloat16, device="npu")
        v_b = torch.randn(2, 8, 64, 64, dtype=torch.bfloat16, device="npu")
        dy_b = torch.randn_like(q_b)
        _run_graph_case(
            lambda q, k, v: torch_npu.npu_fusion_attention_v3(
                q, k, v, head_num=8, input_layout="BNSD", scale=scale, keep_prob=0.9
            )[0],
            q_b, k_b, v_b, dy_b,
            nodrop_fn=lambda q, k, v: torch_npu.npu_fusion_attention_v3(
                q, k, v, head_num=8, input_layout="BNSD", scale=scale, keep_prob=1.0
            )[0],
        )
        _run_graph_case(
            lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.1, scale=scale
            ),
            q_b, k_b, v_b, dy_b,
            nodrop_fn=lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, scale=scale
            ),
        )

        # 3. FAV3 TND
        q_t = torch.randn(64, 8, 64, dtype=torch.bfloat16, device="npu")
        k_t = torch.randn(64, 8, 64, dtype=torch.bfloat16, device="npu")
        v_t = torch.randn(64, 8, 64, dtype=torch.bfloat16, device="npu")
        dy_t = torch.randn_like(q_t)
        seq = torch.tensor([16, 40, 64], dtype=torch.int64, device="cpu")
        _run_graph_case(
            lambda q, k, v, s: torch_npu.npu_fusion_attention_v3(
                q, k, v, head_num=8, input_layout="TND", scale=scale, keep_prob=0.9,
                actual_seq_qlen=s, actual_seq_kvlen=s,
            )[0],
            q_t, k_t, v_t, dy_t,
            extra_args=(seq,),
            nodrop_fn=lambda q, k, v, s: torch_npu.npu_fusion_attention_v3(
                q, k, v, head_num=8, input_layout="TND", scale=scale, keep_prob=1.0,
                actual_seq_qlen=s, actual_seq_kvlen=s,
            )[0],
        )


if __name__ == "__main__":
    run_tests()
