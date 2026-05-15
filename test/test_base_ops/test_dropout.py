import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestDropout(TestCase):

    def _test_dropout_randomness(self, dtype, p):
        input_tensor = torch.randn(12, 12, dtype=dtype).npu()
        dropout_layer = torch.nn.Dropout(p=p)
        output1 = dropout_layer(input_tensor)
        output2 = dropout_layer(input_tensor)

        self.assertNotEqual(output1, output2)

    def _test_dropout_inplace(self, dtype, p):
        input_tensor = torch.randn(12, 12, dtype=dtype).npu()
        dropout_layer = torch.nn.Dropout(p=p, inplace=True)
        output = dropout_layer(input_tensor)

        self.assertTrue(input_tensor is output)

    def _test_dropout_inplace_vs_noninplace(self, dtype, p):
        input_tensor = torch.randn(12, 12, dtype=dtype).npu()

        torch.manual_seed(2)
        dropout_layer = torch.nn.Dropout(p=p, inplace=False)
        output_noninplace = dropout_layer(input_tensor)

        torch.manual_seed(2)
        input_tensor_clone = input_tensor.clone()
        dropout_layer_inplace = torch.nn.Dropout(p=p, inplace=True)
        dropout_layer_inplace(input_tensor_clone)

        self.assertEqual(output_noninplace, input_tensor_clone)

    def _test_dropout_prob_and_scale(self, p):
        input_tensor = torch.ones((100, 100), dtype=torch.float32).npu()
        torch.manual_seed(2)
        output = torch.nn.Dropout(p=p)(input_tensor)

        non_zero_num = torch.count_nonzero(output)
        zero_ratio = 1.0 - non_zero_num / output.numel()
        actual_scale = output.sum() / non_zero_num
        expect_scale = 1.0 / (1.0 - p)

        self.assertTrue(p - 0.05 <= zero_ratio <= p + 0.05)
        self.assertLessEqual(abs(actual_scale - expect_scale), 1e-4)

    def test_dropout_randomness_fp32(self):
        self._test_dropout_randomness(torch.float32, 0.5)

    def test_dropout_randomness_fp16(self):
        self._test_dropout_randomness(torch.float16, 0.5)

    def test_dropout_inplace_fp32(self):
        self._test_dropout_inplace(torch.float32, 0.5)

    def test_dropout_inplace_fp16(self):
        self._test_dropout_inplace(torch.float16, 0.5)

    def test_dropout_inplace_vs_noninplace_fp32(self):
        self._test_dropout_inplace_vs_noninplace(torch.float32, 0.5)

    # Known CANN requirement on A2/A3: fp16 non-inplace dropout and inplace
    # dropout may take different mask-generation paths and cannot guarantee
    # identical masks for the same seed.
    def test_dropout_inplace_vs_noninplace_fp16(self):
        self.skipTest("cann requirement")
        self._test_dropout_inplace_vs_noninplace(torch.float16, 0.5)

    def test_dropout_prob_and_scale_semantics(self):
        self._test_dropout_prob_and_scale(0.1)
        self._test_dropout_prob_and_scale(0.9)

    def test_dropout_p0(self):
        input_tensor = torch.randn(12, 12).npu()
        dropout_layer = torch.nn.Dropout(p=0)
        output = dropout_layer(input_tensor)

        self.assertEqual(output, input_tensor)
        
        dropout_layer_inplace = torch.nn.Dropout(p=0, inplace=True)
        output_inplace = dropout_layer_inplace(input_tensor)

        self.assertTrue(input_tensor is output_inplace)
        self.assertEqual(input_tensor, output_inplace)

    def test_dropout_p1(self):
        input_tensor = torch.randn(12, 12).npu()
        dropout_layer = torch.nn.Dropout(p=1)
        output = dropout_layer(input_tensor)

        self.assertTrue(torch.all(output == 0))
        
        dropout_layer_inplace = torch.nn.Dropout(p=1, inplace=True)
        output_inplace = dropout_layer_inplace(input_tensor)

        self.assertTrue(input_tensor is output_inplace)
        self.assertTrue(torch.all(input_tensor == 0))

     
if __name__ == '__main__':
    run_tests()
