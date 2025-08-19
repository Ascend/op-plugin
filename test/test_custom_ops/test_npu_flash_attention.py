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


def tforward(q, k, v, scale):
    qk = torch.matmul(q, k.permute(0, 1, 3, 2)).mul(scale)
    softmax_res, x_max, x_sum = tsoftmax(qk)
    y = torch.matmul(softmax_res, v)
    return y, softmax_res, x_max, x_sum


def tsoftmax_grad(dp, softmax_res):
    muls = dp * softmax_res
    muls_r = muls.sum(dim=-1, keepdims=True)
    sub_r = dp - muls_r
    res = sub_r * softmax_res
    return res


# pylint:disable = huawei-too-many-arguments
def tbackward(dx, q, k, v, softmax_res, scale):
    drop_res = softmax_res.permute(0, 1, 3, 2)
    dv = torch.matmul(drop_res, dx)
    dp = torch.matmul(dx, v.permute(0, 1, 3, 2))
    softmax_grad_res = (tsoftmax_grad(dp, softmax_res) * scale)
    dq = torch.matmul(softmax_grad_res, k)
    dk = torch.matmul(softmax_grad_res.permute(0, 1, 3, 2), q)
    return dq, dk, dv


class TestNPUFlashAttention(TestCase):
    def supported_op_exec(self, query, key, value, scale):
        qk = torch.matmul(query, key.transpose(2, 3)).mul(scale)
        softmax_res = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        output = torch.matmul(softmax_res, value)
        output = output.transpose(1, 2)
        output = output.reshape(output.shape[0], output.shape[1], -1)
        return output

    # pylint:disable = huawei-too-many-arguments
    def supported_op_exec_with_softmax_out_tnd(self, q, k, v, dx, seqlens_list_q, seqlens_list_k):
        gtype = torch.float32

        # 从输入推导维度
        S1, N1, D = q.shape
        S2, N2, _ = k.shape
        B = len(seqlens_list_q)
        scale = 1 / (D ** 0.5)

        cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
        cu_seqlens_k = get_cu_seqlens(seqlens_list_k)

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

                # 正向golden运算
                outi_golden, softmax_resi, x_maxi, x_sumi = tforward(qi.to(gtype), ki.to(gtype), vi.to(gtype), scale)
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
                dqi_golden, dki_golden, dvi_golden = tbackward(dxi.to(gtype), qi.to(gtype), ki.to(gtype),
                                                               vi.to(gtype), softmax_resi.to(gtype), scale)

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
                       actual_seq_kvlen=None, softmax_layout=""):
        return torch_npu.npu_fusion_attention(
            query, key, value, head_num=head_num, input_layout=input_layout, scale=scale,
            actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen, softmax_layout=softmax_layout)

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
        self.assertRtolEqual(output, attention_score, prec=0.005, prec16=0.005)

    @SupportedDevices(['Ascend910B'])
    @unittest.skip("skip test_npu_flash_attention_tnd now")
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
        self.assertRtolEqual(attention_score_golden, attention_score_npu, prec=0.005, prec16=0.005)
        self.assertRtolEqual(softmax_max_golden, softmax_max_npu, prec=0.005, prec16=0.005)
        self.assertRtolEqual(softmax_sum_golden, softmax_sum_npu, prec=0.005, prec16=0.005)

        # 反向精度比对
        attention_score_npu.backward(dx)
        dq_npu = q.grad
        dk_npu = k.grad
        dv_npu = v.grad

        self.assertRtolEqual(dq_golden, dq_npu, prec=0.005, prec16=0.005)
        self.assertRtolEqual(dk_golden, dk_npu, prec=0.005, prec16=0.005)
        self.assertRtolEqual(dv_golden, dv_npu, prec=0.005, prec16=0.005)


if __name__ == "__main__":
    run_tests()

