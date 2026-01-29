import math
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from einops import rearrange


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


if __name__ == "__main__":
    run_tests()

