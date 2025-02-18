import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class TestBincount(TestCase):
    def cpu_op_exec(self, x):
        output = torch.bincount(x)
        return output.numpy()

    def npu_op_exec(self, x):
        output = torch.bincount(x)
        output = output.cpu()
        return output.numpy()

    def cpu_op_exec_with_weights(self, x, weights):
        output = torch.bincount(x, weights)
        return output.numpy()

    def npu_op_exec_with_weights(self, x, weights):
        output = torch.bincount(x, weights)
        output = output.cpu()
        return output.numpy()

    def test_bincount(self):
        input_param = [
            [np.int8, -1, [0]],
            [np.int8, -1, [np.random.randint(1, 65536)]],
            [np.int16, -1, [0]],
        ]
        weight_dtype = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32]
        shape_format = [[x, y] for x in input_param for y in weight_dtype]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_weight, npu_weight = create_common_tensor([item[1]] + item[0][1:], 0, 100)
            cpu_output = self.cpu_op_exec_with_weights(cpu_input, cpu_weight)
            npu_output = self.npu_op_exec_with_weights(npu_input, npu_weight)
            self.assertRtolEqual(cpu_output, npu_output)

    @skipIfUnsupportMultiNPU(2)
    def test_bincount_device_check(self):
        input_tensor = torch.tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4]).npu()
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).to("npu:1")
        msg = "Expected all tensors to be on the same device, but found at least two devices,"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.bincount(input_tensor, weights=weights)


if __name__ == '__main__':
    run_tests()
