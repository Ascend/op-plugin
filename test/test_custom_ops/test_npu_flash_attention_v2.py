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

if __name__ == "__main__":
    run_tests()
