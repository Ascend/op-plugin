import unittest
import torch
import torch_npu
import numpy as np
from torch_npu import npu_fusion_attention_v3
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


def create_float16_tensor(shape):
    return torch.Tensor(np.random.randn(*shape)).to(torch.half)


def tsoftmax(x, is_fp16=False, dtype='fp16'):
    if is_fp16:
        x = x.float()
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    del x
    del x_sub
    x_max = x_max.cpu()
    x_sum = y.sum(dim=-1, keepdims=True)
    ans = y.div(x_sum)
    if is_fp16 and dtype == 'fp16':
        ans = ans.half()
    if is_fp16 and dtype == 'bf16':
        ans = ans.bfloat16()
    return ans, x_max, x_sum


def tforward(q, k, v, pse, drop_mask, atten_mask, scale, keep_prob, is_fp16=False, dtype='fp16'):
    # 计算时使用传入shape [BNGSD]
    if is_fp16:
        q = q.float()
        k = k.float()
        v = v.float()
    q = q.cpu()
    k = k.cpu()
    qk = torch.matmul(q, k.permute(0, 1, 2, 4, 3)).add(pse).mul(scale)  # (B, N, 1, S, D) * (B, N, 1, D, S) = (B, N, 1, S, S)
    atten_mask = atten_mask.cpu()
    qk = qk + atten_mask * (-10000.0)
    softmax_res, x_max, x_sum = tsoftmax(qk, is_fp16)
    x_max = x_max.cpu()
    x_sum = x_sum.cpu()
    drop_mask = drop_mask.cpu()
    if keep_prob > 0:
        drop_res = softmax_res * drop_mask * (1.0 / keep_prob)
    else:
        drop_res = softmax_res * drop_mask * 0.0
    softmax_res = softmax_res.cpu()
    v = v.cpu()
    y = torch.matmul(drop_res.float(), v)
    if is_fp16 and dtype == 'fp16':
        y = y.half()
    if is_fp16 and dtype == 'bf16':
        y = y.bfloat16()
    return y.cpu(), softmax_res, x_max, x_sum


def test_cpu_fusion_attention_v3(x_, w_, n_, h_):
    # x: B, S, K
    # w: K, 3*H
    B, S, K = x_.shape
    H = w_.shape[1] / 3
    N = n_
    if (N != 0):
        D = int(H / N)
    else:
        D = 0
    G = 1
    pttype = x_.dtype
    qkv = torch.matmul(x_.float(), w_.float())


    def parse_qkv(l_index):
        r_index = l_index + 1
        bsh_res = qkv.flatten()[int(l_index * B * S * H) : int(r_index * B * S * H)]
        bngsd_res = bsh_res.reshape(B, S, G, N, D).permute(0, 3, 2, 1, 4)
        return bngsd_res


    q = parse_qkv(0)
    k = parse_qkv(1)
    v = parse_qkv(2)
    # null pse, atten, drop
    null_pse = torch.from_numpy(np.zeros((B, N, G, S, S))).to(pttype)
    null_drop_mask = torch.from_numpy(np.ones((B, N, G, S, S))).to(torch.uint8)
    null_atten_mask = torch.from_numpy(np.zeros((B, N, G, S, S))).to(pttype)
    default_scale = 1
    default_keep_prob = 1
    return tforward(q, k, v, null_pse, null_drop_mask, null_atten_mask, default_scale, default_keep_prob, is_fp16=True, dtype='fp16')


# kernels case params
b, n, s, d, k_dim = 2, 5, 8192, 128, 5120
h = n * d
head_num = n
input_layout = 'BSH'
head_size = 128
x_shape = (b, s, k_dim)
w_shape = (k_dim, 3 * h)
qkv_shape = (b, s, 3 * h)
dy_shape = (b, s, h)
softmax_max_shape = (b, n, s, 8)
softmax_sum_shape = (b, n, s, 8)
attention_in_shape = (b, n, s, d)


def test_cpu_fusion_attention_v3_02(x_, w_, n_, in_, h_):
    # x: B, S, K
    # w: K, 3*H
    B, S, K = x_.shape
    H = w_.shape[1] / 3
    N = n_
    if (N != 0):
        D = int(H / N)
    else:
        D = 0
    G = 1
    pttype = x_.dtype
    qkv = torch.matmul(x_.float(), w_.float())


    def parse_qkv(l_index):
        r_index = l_index + 1
        bsh_res = qkv.flatten()[int(l_index * B * S * H) : int(r_index * B * S * H)]
        bngsd_res = bsh_res.reshape(B, S, G, N, D).permute(0, 3, 2, 1, 4)
        return bngsd_res


    q = parse_qkv(0)
    k = parse_qkv(1)
    v = parse_qkv(2)
    # null pse, atten, drop
    null_pse = torch.from_numpy(np.zeros((B, N, G, S, S))).to(pttype)
    null_drop_mask = torch.from_numpy(np.ones((B, N, G, S, S))).to(torch.uint8)
    null_atten_mask = torch.from_numpy(np.zeros((B, N, G, S, S))).to(pttype)
    default_scale = 1
    default_keep_prob = 1
    return tforward(q, k, v, null_pse, null_drop_mask, null_atten_mask, default_scale, default_keep_prob, is_fp16=True, dtype='fp16')


class TestFusionAttentionV3(TestCase):


    def supported_op_exec(self, x_, w_, head_num_, input_layout_, head_size_):
        return test_cpu_fusion_attention_v3(x_, w_, head_num_, head_size_)


    def custom_op_exec(self, x_, w_, head_num_, input_layout_, head_size_):
        return npu_fusion_attention_v3(x_, w_, head_num_, input_layout_, head_size_)

    @SupportedDevices(['Ascend910B'])
    def test_npu_fusion_attention_v3(self, device="npu"):
        x = create_float16_tensor(x_shape)
        w = create_float16_tensor(w_shape)
        supported_output = self.supported_op_exec(x, w, head_num, input_layout, head_size)[0]
        supported_output_bsh = supported_output.permute(0, 3, 2, 1, 4).reshape(dy_shape) # convert [BNGSD] into [BSH]
        custom_output = self.custom_op_exec(x.npu(), w.npu(), head_num, input_layout, head_size)[0]
        # WARNING: Precision check will be enabled later
        # self.assertRtolEqual(supported_output_bsh, custom_output)

# kernels case params
b2, n2, s2, d2, k_dim2 = 2, 5, 8192, 128, 5120
h2 = n2 * d2
head_num2 = n2
input_layout2 = 'SBH'
head_size2 = 256
x_shape2 = (s2, b2, k_dim2)
w_shape2 = (k_dim2, 3 * h2)
qkv_shape2 = (s2, b2, 3 * h2)
dy_shape2 = (s2, b2, h2)
softmax_max_shape2 = (b2, n2, s2, 8)
softmax_sum_shape2 = (b2, n2, s2, 8)
attention_in_shape2 = (b2, n2, s2, d2)


class TestFusionAttentionV3_02(TestCase):

    def supported_op_exec(self, x_, w_, head_num_, input_layout_, head_size_):
        return test_cpu_fusion_attention_v3_02(x_, w_, head_num_, input_layout_, head_size_)


    def custom_op_exec(self, x_, w_, head_num_, input_layout_, head_size_):
        return npu_fusion_attention_v3(x_, w_, head_num_, input_layout_, head_size_)

    @SupportedDevices(['Ascend910B'])
    def test_npu_fusion_attention_v3(self, device="npu"):
        x = create_float16_tensor(x_shape2)
        w = create_float16_tensor(w_shape2)
        supported_output = self.supported_op_exec(x, w, head_num2, input_layout2, head_size2)[0]
        supported_output_bsh = supported_output.permute(0, 3, 2, 1, 4).reshape(dy_shape2) # convert [BNGSD] into [BSH]
        custom_output = self.custom_op_exec(x.npu(), w.npu(), head_num2, input_layout2, head_size2)[0]

if __name__ == "__main__":
    run_tests()