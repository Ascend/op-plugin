import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestTo(TestCase):
    def common_op_exec(self, input1, target):
        output = input1.to(target)
        output = output.cpu().numpy()
        return output
    
    def memory_format_exec(self, input1, target):
        output = input1.to(memory_format=target)
        output = output.cpu().numpy()
        return output

    def test_to(self):
        shape_format = [
            [np.float32, 0, [3, 3]],
            [np.float16, 0, [4, 3]],
            [np.int32, 0, [3, 5]],
        ]
        targets = [torch.float16, torch.float32, torch.int32, 'cpu', 'npu']
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -100, 100)
            for target in targets:
                cpu_output = self.common_op_exec(cpu_input1, target)
                npu_output = self.common_op_exec(npu_input1, target)
                self.assertRtolEqual(cpu_output, npu_output)
    
    def test_to_memory_format(self):
        shape_format = [
            [np.float32, 0, [3, 3]],
            [np.float16, 0, [4, 3]],
            [np.int32, 0, [3, 5]],
        ]
        targets = [torch.contiguous_format, torch.preserve_format]
        
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -100, 100)
            for target in targets:
                cpu_output = self.memory_format_exec(cpu_input1, target)
                npu_output = self.memory_format_exec(npu_input1, target)
                self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
