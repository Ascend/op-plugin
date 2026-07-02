import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestBatchNormReduce(TestCase):
    def cuda_op_exec(self, input_data):
        input_float = input_data.to(torch.float32)
        cpu_sum = torch.sum(input_float, dim=[0, 2, 3])
        cpu_square_sum = torch.sum(input_float * input_float, dim=[0, 2, 3])
        return cpu_sum.numpy(), cpu_square_sum.numpy()

    def npu_op_exec(self, *args):
        return torch_npu.batch_norm_reduce(*args)

    def assert_batch_norm_reduce_result(self, cpu_input, npu_input, eps=1e-5, rtol=1e-3, atol=1e-3):
        cpu_sum, cpu_square_sum = self.cuda_op_exec(cpu_input)
        npu_sum, npu_square_sum = self.npu_op_exec(npu_input, eps)

        self.assertEqual((cpu_input.shape[1],), tuple(npu_sum.shape))
        self.assertEqual((cpu_input.shape[1],), tuple(npu_square_sum.shape))
        self.assertEqual(torch.float32, npu_sum.dtype)
        self.assertEqual(torch.float32, npu_square_sum.dtype)
        self.assertRtolEqual(cpu_sum, npu_sum.cpu().numpy(), rtol, atol)
        self.assertRtolEqual(cpu_square_sum, npu_square_sum.cpu().numpy(), rtol, atol)

    def test_batch_norm_reduce_normal_cases(self):
        np.random.seed(1234)
        shape_format = [
            [[np.float32, -1, [2, 3, 12, 12]], 1e-5],
            [[np.float32, -1, [1, 1, 2, 2]], 1e-5],
            [[np.float32, -1, [3, 7, 1, 2]], 1e-5],
            [[np.float16, -1, [2, 3, 12, 12]], 1e-5],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 10)
            self.assert_batch_norm_reduce_result(cpu_input, npu_input, item[-1])

    @SupportedDevices(['Ascend910B'])
    def test_batch_norm_reduce_bfloat16(self):
        torch.manual_seed(1234)
        cpu_input = torch.randn(2, 3, 12, 12, dtype=torch.float32).to(torch.bfloat16)
        npu_input = cpu_input.npu()
        self.assert_batch_norm_reduce_result(cpu_input, npu_input, rtol=4e-3, atol=4e-3)

    def test_batch_norm_reduce_eps_no_effect(self):
        _, npu_input = create_common_tensor([np.float32, -1, [2, 3, 12, 12]], 1, 10)
        npu_sum1, npu_square_sum1 = self.npu_op_exec(npu_input, 1e-5)
        npu_sum2, npu_square_sum2 = self.npu_op_exec(npu_input, 1.0)

        self.assertRtolEqual(npu_sum1.cpu().numpy(), npu_sum2.cpu().numpy())
        self.assertRtolEqual(npu_square_sum1.cpu().numpy(), npu_square_sum2.cpu().numpy())

    def test_batch_norm_reduce_invalid_dim(self):
        npu_input = torch.randn(3, dtype=torch.float32).npu()
        with self.assertRaisesRegex(RuntimeError, "dim input tensor|must more than 1"):
            torch_npu.batch_norm_reduce(npu_input, 1e-5)


if __name__ == "__main__":
    run_tests()
