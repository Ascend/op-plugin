import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestThresholdBackward(TestCase):

    def cpu_op_exec(self, input1, threshold, value):
        input1.requires_grad_()
        output = torch.nn.functional.threshold(input1, threshold, value)
        w = torch.ones_like(output)
        output.backward(w)
        out = input1.grad
        output = output.detach()
        return output.numpy(), out.numpy()

    def npu_op_exec(self, input1, threshold, value):
        input1.requires_grad_()
        output = torch.nn.functional.threshold(input1, threshold, value)
        w = torch.ones_like(output)
        output.backward(w)
        out = input1.grad.to("cpu")
        output = output.detach().to("cpu")
        return output.numpy(), out.numpy()

    def test_threshold_backward_common_shape_format(self):
        shape_format = [
            [[np.float32, 0, (1, 5)], [1.0], [20.0]],
            [[np.float32, 0, (2, 3, 5)], [2.0], [20.0]],
            [[np.float32, 0, (2, 3, 4, 5)], [0], [0]],
            [[np.float32, 3, (1, 5)], [1.0], [20.0]],
            [[np.float32, 3, (2, 3, 5)], [2.0], [20.0]],
            [[np.float32, 3, (2, 3, 4, 5)], [0], [0]],
            [[np.float16, 0, (1, 5)], [1.0], [20.0]],
            [[np.float16, 0, (2, 3, 5)], [2.0], [20.0]],
            [[np.float16, 3, (2, 3, 4, 5)], [0], [0]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 3)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_threshold = npu_threshold = item[1][0]
            cpu_value = npu_value = item[2][0]
            cpu_output1, cpu_output2 = self.cpu_op_exec(cpu_input1, cpu_threshold, cpu_value)
            npu_output1, npu_output2 = self.npu_op_exec(npu_input1, npu_threshold, npu_value)
            self.assertRtolEqual(npu_output1.astype(np.float32), cpu_output1)
            self.assertRtolEqual(npu_output2.astype(np.float32), cpu_output2)

    def test_threshold_backward_broadcast_output_shape(self):
        shape_format = [
            [[np.float32, 0, (9, 34, 48, 25)], [np.float32, 0, (1, 1, 1, 1)], [1.0], 
            [np.float32, 0, (9, 34, 48, 25)], [np.float32, 0, (1, 1, 1, 1)], [0], 
            [np.float32, 0, (9, 34, 48, 25)], [np.float16, 0, (1, 1, 1, 1)], [1.0], 
            [np.float32, 0, (9, 34, 48, 25)], [np.float16, 0, (1, 1, 1, 1)], [0]]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 3)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 3)

            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
                
            threshold = item[2][0]
            cpu_output = torch.ops.aten.threshold_backward(cpu_input1, cpu_input2, threshold=threshold)
            npu_output = torch.ops.aten.threshold_backward(npu_input1, npu_input2, threshold=threshold)
            cpu_output = cpu_output.detach()
            npu_output = npu_output.detach().to("cpu")
            self.assertRtolEqual(npu_output.numpy().astype(np.float32), cpu_output.numpy())

if __name__ == "__main__":
    run_tests()
