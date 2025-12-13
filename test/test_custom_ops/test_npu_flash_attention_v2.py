import math
import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

PER_BLOCK_SIZE = 128
FP8_MAX = 448.0

class TestNPUFlashAttentionV2(TestCase):
    def npu_block_quant(self, tensor, scale):
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

    def calc_quant_scale(self, tensor):
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
    
    def supported_op_exec_quant(self, query, key, value, d_scale_q, d_scale_k, d_scale_v, sparse_params):
        scale = 0.08838
        query = query.to(torch.float32)
        key = key.to(torch.float32)
        value = value.to(torch.float32)
        atten_masks = None
        shape = [1, 8, 256, 256]
        if sparse_params[0] == 0:
            atten_mask_u = np.triu(np.ones(shape), k=sparse_params[2] + 1)
            atten_mask_l = np.tril(np.ones(shape), k=-sparse_params[1] - 1)
            atten_masks = atten_mask_u + atten_mask_l
        atten_mask = torch.tensor(atten_masks).to(torch.float16)
        d_scale_qf = torch.zeros(1, 8, 256, 1, dtype=torch.float32)
        d_scale_kf = torch.zeros(1, 8, 256, 1, dtype=torch.float32)
        d_scale_vf = torch.zeros(1, 8, 256, 1, dtype=torch.float32)
        for i in range(256):
            d_scale_qf[:, :, i, :] = d_scale_q[:, :, i // PER_BLOCK_SIZE, :]
            d_scale_kf[:, :, i, :] = d_scale_k[:, :, i // PER_BLOCK_SIZE, :]
            d_scale_vf[:, :, i, :] = d_scale_v[:, :, i // PER_BLOCK_SIZE, :]
        qk = torch.matmul(query * d_scale_qf, (key * d_scale_kf).transpose(2, 3)).mul(scale)
        qk = qk + atten_mask * (torch.finfo(torch.float32).min)
        max = torch.max(qk, dim=-1, keepdim=True)[0]
        softmax_res = torch.exp(qk - max).to(torch.float8_e4m3fn).to(torch.float32)
        sum = torch.sum(softmax_res, dim=-1, keepdim=True)
        attention_out = torch.matmul(softmax_res, value * d_scale_vf)
        attention_out_res = attention_out / sum
        return attention_out_res
    
    def custom_op_exec_quant(self, query, key, value, d_scale_q, d_scale_k, d_scale_v, sparse_params):
        scale = 0.08838
        atten_mask = None
        if sparse_params[0] == 0:
            shape = [1, 8, 256, 256]
            atten_mask_u = np.triu(np.ones(shape), k=sparse_params[1] + 1)
            atten_mask_l = np.tril(np.ones(shape), k=-sparse_params[2] - 1)
            atten_masks = atten_mask_u + atten_mask_l
            atten_mask = torch.tensor(atten_masks).to(torch.float16).bool().npu()
        return torch_npu.npu_fusion_attention_v2(
            query, key, value, head_num=8, input_layout="BNSD", scale=scale, sparse_mode=sparse_params[0],
            atten_mask=atten_mask, d_scale_q=d_scale_q, d_scale_k=d_scale_k, d_scale_v=d_scale_v,
            pre_tokens=sparse_params[1], next_tokens=sparse_params[2])

    def supported_op_exec(self, query, key, value, drop_mask=None, keep_prob=1.0):
        scale = 0.08838
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

    def custom_op_exec(self, query, key, value, keep_prob=1.0):
        scale = 0.08838
        return torch_npu.npu_fusion_attention_v2(
            query, key, value, head_num=32, input_layout="BSH", scale=scale, keep_prob=keep_prob)

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

        q_npu = self.trans_BNSD2BSH(query).npu()
        k_npu = self.trans_BNSD2BSH(key).npu()
        v_npu = self.trans_BNSD2BSH(value).npu()
        output = self.supported_op_exec(query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)).to(torch.float16)
        attention_score, softmax_max, softmax_sum, softmax_out, seed, offset, numels = self.custom_op_exec(q_npu, k_npu, v_npu)
        self.assertRtolEqual(output, attention_score)

    @SupportedDevices(['Ascend910B'])
    def test_npu_flash_attention_v2_with_dropmask(self, device="npu"):
        query = torch.randn(1, 32, 256, 128, dtype=torch.float16)
        key = torch.randn(1, 32, 256, 128, dtype=torch.float16)
        value = torch.randn(1, 32, 256, 128, dtype=torch.float16)
        keep_prob = 0.9
        drop_mask = self.get_drop_mask(query, 1, 32, 256, 256, seed=2, gen_p=1-keep_prob)

        q_npu = self.trans_BNSD2BSH(query).npu()
        k_npu = self.trans_BNSD2BSH(key).npu()
        v_npu = self.trans_BNSD2BSH(value).npu()
        output = self.supported_op_exec(query.to(torch.float32), key.to(torch.float32), value.to(torch.float32),
                                        drop_mask, keep_prob).to(torch.float16)
        attention_score, _, _, _, _, _, _ = self.custom_op_exec(q_npu, k_npu, v_npu, keep_prob)
        self.assertRtolEqual(output, attention_score)
    
    @SupportedDevices(['Ascend910_95'])
    def test_npu_flash_attention_v2_with_fp8(self, device="npu"):
        query = torch.randn(1, 8, 256, 256, dtype=torch.float16)
        key = torch.randn(1, 8, 256, 256, dtype=torch.float16)
        value = torch.randn(1, 8, 256, 256, dtype=torch.float16)
        scale_q = self.calc_quant_scale(query).to(torch.float32)
        scale_k = self.calc_quant_scale(key).to(torch.float32)
        scale_v = self.calc_quant_scale(value).to(torch.float32)
        d_scale_q = 1 / scale_q
        d_scale_k = 1 / scale_k
        d_scale_v = 1 / scale_v
        query_fp8 = self.npu_block_quant(query, scale_q).to(torch.float8_e4m3fn)
        key_fp8 = self.npu_block_quant(key, scale_k).to(torch.float8_e4m3fn)
        value_fp8 = self.npu_block_quant(value, scale_v).to(torch.float8_e4m3fn)
        sparse_params = [0, 128, 128]
        output = self.supported_op_exec_quant(query_fp8.to(torch.float32), key_fp8.to(torch.float32), value_fp8.to(torch.float32),
                                              d_scale_q, d_scale_k, d_scale_v, sparse_params)
        fa_result = self.custom_op_exec_quant(query_fp8.npu(), key_fp8.npu(), value_fp8.npu(), d_scale_q.npu(), d_scale_k.npu(),
                                              d_scale_v.npu(), sparse_params)
        self.assertRtolEqual(output.half(), fa_result[0], prec=0.01, prec16=0.01)

if __name__ == "__main__":
    run_tests()
