import unittest

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestHansCompress(TestCase):

    @unittest.skip("skip test_hans_compress now")
    def test_hans_compress(self):
        data_shape = (4096, 512)
        statistic = True
        dtype_list = [torch.float32, torch.bfloat16, torch.float16]
        reshuff_list = [False, True]
        for dtype in dtype_list:
            for reshuff in reshuff_list:
                input_tensor = torch.randn(data_shape, dtype=dtype).npu()
                recover = torch.zeros(data_shape, dtype=dtype).npu()
                pdf = torch.zeros(256, dtype=torch.int32).npu()
                mantissa_numel = input_tensor.numel() * (input_tensor.element_size() - 1)
                exp_numel = input_tensor.numel()
                mantissa =  torch.zeros(mantissa_numel // input_tensor.element_size(), dtype=input_tensor.dtype).npu()
                fixed = torch.zeros(exp_numel, dtype=input_tensor.dtype).npu()
                var = torch.zeros(exp_numel, dtype=input_tensor.dtype).npu()
                pdf, mantissa, fixed, var = torch_npu.npu_hans_encode(
                    input_tensor, statistic, reshuff, out=(pdf, mantissa, fixed, var))
                recover = torch_npu.npu_hans_decode(mantissa, fixed, var, pdf, reshuff, out=recover)
                self.assertEqual(input_tensor, recover,
                    message=f"[ERROR] Hans decompress falied. dtype:{dtype} reshuff: {reshuff}")

if __name__ == "__main__":
    run_tests()