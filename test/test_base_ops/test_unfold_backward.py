import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestUnfoldBackward(TestCase):

    def cpu_op_exec(self, input1, input_sizes, dim, size, step):
        output = torch.ops.aten.unfold_backward(input1, input_sizes, dim, size, step)
        out_cpu = output.numpy()
        return out_cpu

    def npu_op_exec(self, input1, input_sizes, dim, size, step):
        output = torch.ops.aten.unfold_backward(input1, input_sizes, dim, size, step)
        out_npu = output.cpu().numpy()
        return out_npu

    def test_unfold_backward_shape_format(self):
        shape_format = [
            [[np.float32, 0, (1, 3, 3, 320, 658, 20)], [1, 3, 3, 658, 658], 3, 20, 2],
            [[np.float32, 0, (1, 3, 3, 658, 320, 20)], [1, 3, 3, 658, 658], 4, 20, 2],
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -5, 5)
            out_cpu = self.cpu_op_exec(cpu_input, item[1], item[2], item[3], item[4])
            out_npu = self.npu_op_exec(npu_input, item[1], item[2], item[3], item[4])
            self.assertRtolEqual(out_cpu, out_npu)

    def test_unfold_backward_empty_output(self):
        # input_sizes with a zero dim yields an empty output. Native CPU cannot
        # serve as a reference here (it crashes with setStorage on zero dims),
        # so this test verifies the NPU short-circuit: it must return the empty
        # result with the right shape/dtype instead of hitting the CANN
        # empty-tensor (EZ1001) error.
        npu_input = torch.randn((0, 20)).npu()
        out_npu = torch.ops.aten.unfold_backward(npu_input, [0, 20], 0, 20, 2)
        self.assertEqual(out_npu.shape, (0, 20))
        self.assertEqual(out_npu.numel(), 0)
        self.assertEqual(out_npu.device.type, "npu")


if __name__ == "__main__":
    run_tests()
