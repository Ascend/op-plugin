import math
import unittest
import torch
import torch_npu
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUFusedFloydAttention(TestCase):
    def truncated_normal(self, mean, std, min, max, size):
        x = torch.normal(mean, std, size)
        x = torch.where((x < min) | (x > max), torch.tensor(0.0), x)
        return x

    def tsoftmax(self, x):
        x_max = torch.max(x, dim=-1, keepdims=True)[0]
        x_sub = x.sub(x_max)
        y = torch.exp(x_sub)
        x_sum = y.sum(dim=-1, keepdims=True)
        ans = y.div(x_sum)
        return ans, x_max, x_sum

    def tsoftmax_grad(self, dp, softmax_res):
        muls = dp * softmax_res
        muls_r = muls.sum(dim=-1, keepdims=True)
        sub_r = dp - muls_r
        res = sub_r * softmax_res
        return res

    def simple_softmax(self, src, max, sum):
        dst = torch.exp(src - max) / sum
        return dst

    def floyd_attn_forward_manual_simple(self, Q, K1, K2, V1, V2, scale, atten_mask):
        s = torch.einsum('bhikc,bhijc->bhikj', Q, K1) + torch.einsum('bhikc,bhjkc->bhikj', Q, K2)
        if atten_mask != None:
            s = s + atten_mask.bool() * (-4000000000.0)
        p, x_max, x_sum = self.tsoftmax(s * scale)
        output = torch.einsum('bhikj,bhijc->bhikc', p, V1) + torch.einsum('bhikj,bhjkc->bhikc', p, V2)
        return output, x_max, x_sum

    def floyd_attn_backward_manual_simple(self, Q, K1, K2, V1, V2, x_max, x_sum, grad, output, scale, atten_mask):
        s = torch.einsum('bhikc,bhijc->bhikj', Q, K1) + torch.einsum('bhikc,bhjkc->bhikj', Q, K2)
        dp = torch.einsum('bhikc,bhijc->bhikj', grad, V1)  + torch.einsum('bhikc,bhjkc->bhikj', grad, V2)

        if atten_mask != None:
            s = s + atten_mask.bool() * (-4000000000.0)

        p = self.simple_softmax(s * scale, x_max, x_sum)
        ds = p * (dp - (grad * output).sum(dim=-1, keepdim=True)) * scale
        
        dQ = torch.einsum('bhikj,bhijc->bhikc', ds, K1) + torch.einsum('bhikj,bhjkc->bhikc', ds, K2)
        dK1 = torch.einsum('bhikj,bhikc->bhijc', ds, Q)
        dK2 = torch.einsum('bhikj,bhikc->bhjkc', ds, Q)
        dV1 = torch.einsum('bhikj,bhikc->bhijc', p, grad)
        dV2 = torch.einsum('bhikj,bhikc->bhjkc', p, grad)
        return dQ, dK1, dV1, dK2, dV2

    @SupportedDevices(['Ascend910B'])
    @unittest.skip("skip") # CI版本不支持
    def test_floyd_attention(self):
        B, N, S1, S2, S3, D = 1, 1, 16, 256, 256, 64
        Q = self.truncated_normal(0.0, 1, -10, 10, (B, N, S1, S2, D)).to(torch.bfloat16).npu()
        K1 = self.truncated_normal(0.0, 1, -10, 10, (B, N, S1, S3, D)).to(torch.bfloat16).npu()
        K2 = self.truncated_normal(0.0, 1, -10, 10, (B, N, S3, S2, D)).to(torch.bfloat16).npu()
        V1 = self.truncated_normal(0.0, 1, -10, 10, (B, N, S1, S3, D)).to(torch.bfloat16).npu()
        V2 = self.truncated_normal(0.0, 1, -10, 10, (B, N, S3, S2, D)).to(torch.bfloat16).npu()
        grad = self.truncated_normal(0.0, 1, -10, 10, (B, N, S1, S2, D)).to(torch.bfloat16).npu()

        atten_mask = torch.randint(0, 2, [B, 1, S1, 1, S3]).to(torch.bool).npu()
        scale = 1.0/math.sqrt(D)
        gtype = torch.float32
        atten_mask = torch.zeros([B, 1, S1, 1, S3]).to(torch.bool).npu()
        output, x_max, x_sum = self.floyd_attn_forward_manual_simple(Q.to(gtype), K1.to(gtype), K2.to(gtype), 
                                                            V1.to(gtype), V2.to(gtype), scale, atten_mask)

        dQ, dK1, dV1, dK2, dV2 = self.floyd_attn_backward_manual_simple(Q.to(gtype), K1.to(gtype), K2.to(gtype),
                                                    V1.to(gtype), V2.to(gtype), x_max.to(gtype),
                                                    x_sum.to(gtype), grad.to(gtype), output.to(gtype), scale, atten_mask)

        x_max = x_max.reshape(B, N, S1, S2, 1).broadcast_to(B, N, S1, S2, 8)
        x_sum = x_sum.reshape(B, N, S1, S2, 1).broadcast_to(B, N, S1, S2, 8)
        Q.requires_grad = True
        K1.requires_grad = True
        V1.requires_grad = True
        K2.requires_grad = True
        V2.requires_grad = True
        x_max_npu, x_sum_npu, output_npu = torch_npu.npu_fused_floyd_attention(
            Q,
            K1,
            V1,
            K2,
            V2,
            atten_mask = atten_mask,
            scale_value = scale
        )
        self.assertRtolEqual(output.cpu(), output_npu.cpu().float(), 0.005, 0.005)
        self.assertRtolEqual(x_max.cpu(), x_max_npu.cpu())
        self.assertRtolEqual(x_sum.cpu(), x_sum_npu.cpu())

        output_npu.backward(grad)
        self.assertRtolEqual(Q.grad.cpu().float(), dQ.cpu().float(), 0.005, 0.005)
        self.assertRtolEqual(K1.grad.cpu().float(), dK1.cpu().float(), 0.005, 0.005)
        self.assertRtolEqual(V1.grad.cpu().float(), dV1.cpu().float(), 0.005, 0.005)
        self.assertRtolEqual(K2.grad.cpu().float(), dK2.cpu().float(), 0.005, 0.005)
        self.assertRtolEqual(V2.grad.cpu().float(), dV2.cpu().float(), 0.005, 0.005)

    @SupportedDevices(['Ascend910B'])
    @unittest.skip("skip") # CI版本不支持
    def test_floyd_attention_grad(self):
        B, N, S1, S2, S3, D = 1, 1, 16, 256, 256, 64
        Q = self.truncated_normal(0.0, 1, -10, 10, (B, N, S1, S2, D)).to(torch.bfloat16).npu()
        K1 = self.truncated_normal(0.0, 1, -10, 10, (B, N, S1, S3, D)).to(torch.bfloat16).npu()
        K2 = self.truncated_normal(0.0, 1, -10, 10, (B, N, S3, S2, D)).to(torch.bfloat16).npu()
        V1 = self.truncated_normal(0.0, 1, -10, 10, (B, N, S1, S3, D)).to(torch.bfloat16).npu()
        V2 = self.truncated_normal(0.0, 1, -10, 10, (B, N, S3, S2, D)).to(torch.bfloat16).npu()
        grad = self.truncated_normal(0.0, 1, -10, 10, (B, N, S1, S2, D)).to(torch.bfloat16).npu()

        atten_mask = torch.randint(0, 2, [B, 1, S1, 1, S3]).to(torch.bool).npu()
        scale = 1.0/math.sqrt(D)
        gtype = torch.float32
        atten_mask = torch.zeros([B, 1, S1, 1, S3]).to(torch.bool).npu()

        output, x_max, x_sum = self.floyd_attn_forward_manual_simple(Q.to(gtype), K1.to(gtype), K2.to(gtype), 
                                                            V1.to(gtype), V2.to(gtype), scale, atten_mask)

        dQ, dK1, dV1, dK2, dV2 = self.floyd_attn_backward_manual_simple(Q.to(gtype), K1.to(gtype), K2.to(gtype),
                                                    V1.to(gtype), V2.to(gtype), x_max.to(gtype),
                                                    x_sum.to(gtype), grad.to(gtype), output.to(gtype), scale, atten_mask)

        x_max = x_max.reshape(B, N, S1, S2, 1).broadcast_to(B, N, S1, S2, 8)
        x_sum = x_sum.reshape(B, N, S1, S2, 1).broadcast_to(B, N, S1, S2, 8)
        dq, dk0, dv0, dk1, dv1 = torch_npu.npu_fused_floyd_attention_backward(
            grad,
            Q,
            K1,
            V1,
            K2,
            V2,
            output.reshape(B, N , S1, S2, D).to(torch.bfloat16).npu(),
            x_max.npu(),
            x_sum.npu(),
            atten_mask = atten_mask,
            scale_value = scale
        )
        self.assertRtolEqual(dQ.cpu(), dq.cpu().float(), 0.005, 0.005)
        self.assertRtolEqual(dK1.cpu(), dk0.cpu().float(), 0.005, 0.005)
        self.assertRtolEqual(dV1.cpu(), dv0.cpu().float(), 0.005, 0.005)
        self.assertRtolEqual(dK2.cpu(), dk1.cpu().float(), 0.005, 0.005)
        self.assertRtolEqual(dV2.cpu(), dv1.cpu().float(), 0.005, 0.005)


if __name__ == '__main__':
    run_tests()
