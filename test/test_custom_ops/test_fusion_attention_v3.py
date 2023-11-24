import torch_npu
import torch
import numpy as np
from torch_npu import npu_fusion_attention_grad_v3


def create_float16_tensor(shape):
    return torch.Tensor(np.random.randn(*shape)).to(torch.half).npu()


def create_float32_tensor(shape):
    return torch.Tensor(np.random.randn(*shape)).to(torch.float32).npu()


def create_bfloat16_tensor(shape):
    return torch.Tensor(np.random.randn(*shape)).to(torch.bfloat16).npu()


# kernels case params
B, N, S, D, K = 2, 5, 8192, 128, 5120
H = N * D
head_num = N
input_layout = 'BSH'

x_shape = (B, S, K)
w_shape = (H, 3 * H)
qkv_shape = (B, S, 3 * H)
dy_shape = (B, S, H)
softmax_max_shape = (B, N, S, 8)
softmax_sum_shape = (B, N, S, 8)
attention_in_shape = (B, N, S, D)


def test_fusion_attention_grad_v3_fp16():
    x = create_float16_tensor(x_shape)
    w = create_float16_tensor(w_shape)
    qkv = create_float16_tensor(qkv_shape)
    dy = create_float16_tensor(dy_shape)
    softmax_max = create_float32_tensor(softmax_max_shape)
    softmax_sum = create_float32_tensor(softmax_sum_shape)
    attention_in = create_float16_tensor(attention_in_shape)
    res = npu_fusion_attention_grad_v3(x, w, qkv, dy, softmax_max, softmax_sum, attention_in, head_num, input_layout)


def test_fusion_attention_grad_v3_bfp16():
    x = create_bfloat16_tensor(x_shape)
    w = create_bfloat16_tensor(w_shape)
    qkv = create_bfloat16_tensor(qkv_shape)
    dy = create_bfloat16_tensor(dy_shape)
    softmax_max = create_float32_tensor(softmax_max_shape)
    softmax_sum = create_float32_tensor(softmax_sum_shape)
    attention_in = create_bfloat16_tensor(attention_in_shape)
    res = npu_fusion_attention_grad_v3(x, w, qkv, dy, softmax_max, softmax_sum, attention_in, head_num, input_layout)


# test case1: fag_v3 in fp16 precision
test_fusion_attention_grad_v3_fp16()
# test case2: fag_v3 in bfp16 precision
test_fusion_attention_grad_v3_bfp16()
