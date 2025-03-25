import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestGather(TestCase):
    def cpu_op_exec(self, input1, dim, index, sparse_grad=False):
        input1.requires_grad = True
        output = torch.gather(input1, dim, index, sparse_grad=sparse_grad)
        output.backward(torch.ones_like(output))
        output = output.detach().numpy()
        return output, input1.grad

    def npu_op_exec(self, input1, dim, index, sparse_grad=False):
        input1.requires_grad = True
        output = torch.gather(input1, dim, index, sparse_grad=sparse_grad)
        output.backward(torch.ones_like(output))
        output = output.to("cpu")
        output = output.detach().numpy()
        return output, input1.grad

    def test_gather_backward_shape_format(self):
        shape_format = [
            [[np.float32, 0, (3, 3, 5)], -3, torch.LongTensor([[[0, 1, 2, 0, 2], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
                                                               [[1, 2, 2, 2, 2], [0, 0, 0, 0, 0], [2, 2, 2, 2, 2]]])]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2])
            npu_idx = item[2].to("npu")
            npu_output = self.npu_op_exec(npu_input1, item[1], npu_idx)
            self.assertRtolEqual(cpu_output, npu_output)


    def test_gather_gather_saprse_grad(self):
        shape_format = [
            [[np.float32, 0, (3, 3, 5)], -3, torch.LongTensor([[[0, 1, 2, 0, 2], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
                                                               [[1, 2, 2, 2, 2], [0, 0, 0, 0, 0], [2, 2, 2, 2, 2]]])]
        ]
        for item in shape_format:
            _, npu_input1 = create_common_tensor(item[0], -100, 100)
            npu_idx = item[2].to("npu")
            with self.assertRaises(RuntimeError) as cm:
                self.npu_op_exec(npu_input1, item[1], npu_idx, True)
            exception = cm.exception
            self.assertTrue("npu_gather_backward not support sparse" in str(exception))


if __name__ == "__main__":
    run_tests()
