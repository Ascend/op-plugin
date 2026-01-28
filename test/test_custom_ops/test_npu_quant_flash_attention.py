import math
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

PER_BLOCK_Q_SIZE = 128
PER_BLOCK_KV_SIZE = 256
FP8_MAX = 448.0

class TestNPUQuantFlashAttentionV2(TestCase):
    def npu_block_quant(self, tensor, scale, PER_BLOCK_SIZE=128):
        dim1 = tensor.shape[0]
        dim2 = tensor.shape[1]
        dim3 = tensor.shape[2]
        dim4 = tensor.shape[3]
        quanted_tensor = torch.zeros([dim1, dim2, dim3, dim4]).to(torch.float32)
        for b in range(dim1):
            for n in range(dim2):
                for s in range(0, dim3, PER_BLOCK_SIZE):
                    s_start = s // PER_BLOCK_SIZE * PER_BLOCK_SIZE
                    s_end = min(s // PER_BLOCK_SIZE * PER_BLOCK_SIZE + PER_BLOCK_SIZE, dim3)
                    quanted_tensor[b, n, s_start:s_end, :] = tensor[b, n, s_start:s_end, :] * scale[
                        b, n, s // PER_BLOCK_SIZE, 0]
        return quanted_tensor

    def calc_quant_scale(self, tensor, PER_BLOCK_SIZE=128):
        dim1 = tensor.shape[0]
        dim2 = tensor.shape[1]
        dim3 = tensor.shape[2]
        dim4 = tensor.shape[3]
        scale_for_tensor = torch.ones([dim1, dim2, math.ceil(dim3 / PER_BLOCK_SIZE), 1]).to(torch.float32)
        for b in range(dim1):
            for n in range(dim2):
                for s in range(0, dim3, PER_BLOCK_SIZE):
                    s_start = s // PER_BLOCK_SIZE * PER_BLOCK_SIZE
                    s_end = min(s // PER_BLOCK_SIZE * PER_BLOCK_SIZE + PER_BLOCK_SIZE, dim3)
                    chunk = tensor[b, n, s_start:s_end, :]
                    max_val = torch.max(torch.abs(chunk))
                    epsilon = 1e-8
                    scale_for_tensor[b, n, s // PER_BLOCK_SIZE, 0] = FP8_MAX / (max_val + epsilon)
        return scale_for_tensor
    
    def supported_op_exec_quant(self, query, key, value, d_scale_q, d_scale_k, d_scale_v):
        scale = 0.08838
        query = query.to(torch.float32)
        key = key.to(torch.float32)
        value = value.to(torch.float32)
        d_scale_qf = torch.zeros(1, 8, 256, 1, dtype=torch.float32)
        d_scale_kf = torch.zeros(1, 8, 256, 1, dtype=torch.float32)
        d_scale_vf = torch.zeros(1, 8, 256, 1, dtype=torch.float32)
        for i in range(256):
            d_scale_qf[:, :, i, :] = d_scale_q[:, :, i // PER_BLOCK_Q_SIZE, :]
            d_scale_kf[:, :, i, :] = d_scale_k[:, :, i // PER_BLOCK_KV_SIZE, :]
            d_scale_vf[:, :, i, :] = d_scale_v[:, :, i // PER_BLOCK_KV_SIZE, :]
        qk = torch.matmul(query * d_scale_qf, (key * d_scale_kf).transpose(2, 3)).mul(scale)
        max = torch.max(qk, dim=-1, keepdim=True)[0]
        softmax_res = torch.exp(qk - max).to(torch.float8_e4m3fn).to(torch.float32)
        sum = torch.sum(softmax_res, dim=-1, keepdim=True)
        attention_out = torch.matmul(softmax_res, value * d_scale_vf)
        attention_out_res = attention_out / sum
        return attention_out_res
    
    def custom_op_exec_quant(self, query, key, value, d_scale_q, d_scale_k, d_scale_v):
        scale = 0.08838
        return torch_npu.npu_quant_fusion_attention(
            query, key, value, head_num=8, input_layout="BNSD", scale=scale,
            d_scale_q=d_scale_q, d_scale_k=d_scale_k, d_scale_v=d_scale_v)

    
    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_quant_flash_attention_with_fp8(self, device="npu"):
        query = torch.randn(1, 8, 256, 128, dtype=torch.float16)
        key = torch.randn(1, 8, 256, 128, dtype=torch.float16)
        value = torch.randn(1, 8, 256, 128, dtype=torch.float16)
        scale_q = self.calc_quant_scale(query, 128).to(torch.float32)
        scale_k = self.calc_quant_scale(key, 256).to(torch.float32)
        scale_v = self.calc_quant_scale(value, 256).to(torch.float32)
        d_scale_q = 1 / scale_q
        d_scale_k = 1 / scale_k
        d_scale_v = 1 / scale_v
        query_fp8 = self.npu_block_quant(query, scale_q, 128).to(torch.float8_e4m3fn)
        key_fp8 = self.npu_block_quant(key, scale_k, 256).to(torch.float8_e4m3fn)
        value_fp8 = self.npu_block_quant(value, scale_v, 256).to(torch.float8_e4m3fn)
        output = self.supported_op_exec_quant(query_fp8.to(torch.float32), key_fp8.to(torch.float32), value_fp8.to(torch.float32),
                                              d_scale_q, d_scale_k, d_scale_v)
        fa_result = self.custom_op_exec_quant(query_fp8.npu(), key_fp8.npu(), value_fp8.npu(), d_scale_q.npu(), d_scale_k.npu(),
                                              d_scale_v.npu())
        self.assertRtolEqual(output.half(), fa_result[0], prec=0.01, prec16=0.01)

if __name__ == "__main__":
    run_tests()
