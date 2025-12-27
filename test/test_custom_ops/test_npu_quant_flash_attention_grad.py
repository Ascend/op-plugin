import math
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


def tsoftmax_grad(dp, softmax_res):
    muls = dp * softmax_res
    muls_r = muls.sum(dim=-1, keepdims=True)
    sub_r = dp - muls_r
    res = sub_r * softmax_res
    return res


def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    ans = y.div(x_sum)
    return ans, x_max, x_sum


class TestNPUQuantFlashAttentionV2(TestCase):
    def supported_op_exec(self, query, key, value, dy, drop_mask=None, keep_prob=1.0):
        scale = 0.08838
        qk = torch.matmul(query, key.transpose(2, 3)).mul(scale)
        softmax_res, x_max, x_sum = tsoftmax(qk.to(torch.float32))
        dp = torch.matmul(dy, value.transpose(2, 3))
        if drop_mask == None or len(drop_mask.shape) == 0:
            drop_res = softmax_res
            dp_drop = dp
        else:
            drop_res = softmax_res.mul(drop_mask).mul(1.0 / keep_prob)
            dp_drop = dp * drop_mask * (1.0 / keep_prob)
        y = torch.matmul(drop_res, value)
        dv = torch.matmul(drop_res.transpose(2, 3), dy)
        softmax_grad_res = (tsoftmax_grad(dp_drop, softmax_res) * scale)
        dq = torch.matmul(softmax_grad_res, key)
        dk = torch.matmul(softmax_grad_res.transpose(2, 3), query)       
        dq = dq.transpose(1, 2)
        dq = dq.reshape(dq.shape[0], dq.shape[1], -1)
        dk = dk.transpose(1, 2)
        dk = dk.reshape(dk.shape[0], dk.shape[1], -1)
        dv = dv.transpose(1, 2)
        dv = dv.reshape(dv.shape[0], dv.shape[1], -1)
        return y, softmax_res, x_max, x_sum, dq, dk, dv

    def custom_op_exec(self, query, key, value, dy, softmax_max, softmax_sum, attention_in, keep_prob=1.0, numels=0, seed=2):
        scale = 0.08838
        return torch_npu.npu_fusion_attention_grad_v2(
            query, key, value, dy, head_num=32, input_layout="BSH", softmax_max=softmax_max, softmax_sum=softmax_sum,
            attention_in=attention_in, scale_value=scale, keep_prob=keep_prob, numels=numels, seed=seed)

    def trans_BNSD2BSH(self, tensor: torch.Tensor):
        tensor = torch.transpose(tensor, 1, 2)
        tensor = torch.reshape(tensor, (tensor.shape[0], tensor.shape[1], -1))
        return tensor
    
    def get_drop_mask(self, q, B, N1, S1, S2, seed=2, gen_p=0.2):
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.manual_seed(seed)
        drop_mask_uint8 = torch_npu._npu_dropout_gen_mask(q.npu(), [B, N1, S1, S2], p=gen_p, seed=seed, offset=0,
                                                          parallel=True, sync=False)
        drop_mask_bit_np = np.unpackbits(drop_mask_uint8.cpu().numpy(), count=B*N1*S1*S2, bitorder='little')
        drop_mask_bit = torch.from_numpy(drop_mask_bit_np).reshape([B, N1, S1, S2])
        drop_mask_bit = drop_mask_bit.detach().clone().to(torch.uint8)
        return drop_mask_bit.cpu()

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_v2(self, device="npu"):
        query = torch.randn(1, 32, 128, 128, dtype=torch.float16)
        key = torch.randn(1, 32, 128, 128, dtype=torch.float16)
        value = torch.randn(1, 32, 128, 128, dtype=torch.float16)
        dy = torch.randn(1, 32, 128, 128, dtype=torch.float16)

        q_npu = self.trans_BNSD2BSH(query).npu()
        k_npu = self.trans_BNSD2BSH(key).npu()
        v_npu = self.trans_BNSD2BSH(value).npu()
        dy_npu = self.trans_BNSD2BSH(dy).npu()
        out, softmax_res, x_max, x_sum, dq_cpu, dk_cpu, dv_cpu = self.supported_op_exec(query.to(torch.float32), key.to(torch.float32), value.to(torch.float32), dy.to(torch.float32))
        x_max = x_max.expand(1, 32, 128, 8).npu()
        x_sum = x_sum.expand(1, 32, 128, 8).npu()
        out_npu = self.trans_BNSD2BSH(out).to(torch.float16).npu()
        dq, dk, dv, dpse, dq_rope, dk_rope, dsink = self.custom_op_exec(q_npu, k_npu, v_npu, dy_npu, x_max, x_sum, out_npu)
        self.assertRtolEqual(dq_cpu, dq.to(torch.float32), prec=0.005, prec16=0.005)

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_v2_with_dropmask(self, device="npu"):
        query = torch.randn(1, 32, 256, 128, dtype=torch.float16)
        key = torch.randn(1, 32, 256, 128, dtype=torch.float16)
        value = torch.randn(1, 32, 256, 128, dtype=torch.float16)
        dy = torch.randn(1, 32, 256, 128, dtype=torch.float16)
        keep_prob = 0.9
        numels = 1 * 32 * 256 * 256
        drop_mask = self.get_drop_mask(query, 1, 32, 256, 256, seed=2, gen_p=1-keep_prob)

        q_npu = self.trans_BNSD2BSH(query).npu()
        k_npu = self.trans_BNSD2BSH(key).npu()
        v_npu = self.trans_BNSD2BSH(value).npu()
        dy_npu = self.trans_BNSD2BSH(dy).npu()
        out, softmax_res, x_max, x_sum, dq_cpu, dk_cpu, dv_cpu = self.supported_op_exec(
            query.to(torch.float32),key.to(torch.float32), value.to(torch.float32),
            dy.to(torch.float32), drop_mask, keep_prob)
        x_max = x_max.expand(1, 32, 256, 8).npu()
        x_sum = x_sum.expand(1, 32, 256, 8).npu()
        out_npu = self.trans_BNSD2BSH(out).to(torch.float16).npu()
        dq, dk, dv, dpse, dq_rope, dk_rope, dsink = self.custom_op_exec(q_npu, k_npu, v_npu, dy_npu, x_max, x_sum,
                                                                 out_npu, keep_prob, numels)
        self.assertRtolEqual(dq_cpu, dq.to(torch.float32), prec=0.005, prec16=0.005)

if __name__ == "__main__":
    run_tests()
