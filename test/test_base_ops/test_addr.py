import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAddr(TestCase):
    def cpu_op_exec(self, input1, vec1, vec2, beta, alpha):
        output = torch.addr(input1, vec1, vec2, beta=beta, alpha=alpha)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, vec1, vec2, beta, alpha):
        output = torch.addr(input1, vec1, vec2, beta=beta, alpha=alpha)
        output = output.to("cpu")
        output = output.numpy()
        return output

    # pylint:disable = huawei-too-many-arguments
    def npu_op_exec_out(self, input1, input2, vec1, vec2, beta, alpha):
        torch.addr(input1, vec1, vec2, beta=beta, alpha=alpha, out=input2)
        output = input2.to("cpu")
        output = output.numpy()
        return output

    def test_addr_common_shape_format(self):
        shape_format = [
            [[np.bool_, 0, (5, 3)], [np.bool_, 0, (5)], [np.bool_, 0, (3)]],
            [[np.bool_, 0, (5, 3)], [np.int32, 0, (5)], [np.int32, 0, (3)]],
            [[np.bool_, 0, (5, 3)], [np.float32, 0, (5)], [np.float32, 0, (3)]],
            [[np.bool_, 0, (5, 3)], [np.int32, 0, (5)], [np.float32, 0, (3)]],
            [[np.int32, 0, (5, 3)], [np.int32, 0, (5)], [np.int32, 0, (3)]],
            [[np.int32, 0, (5, 3)], [np.int32, 0, (5)], [np.float32, 0, (3)]],
            [[np.int32, 0, (5, 3)], [np.float32, 0, (5)], [np.float32, 0, (3)]],
            [[np.int32, 0, (5, 3)], [np.bool_, 0, (5)], [np.float32, 0, (3)]],
            [[np.float32, 0, (5, 3)], [np.float32, 0, (5)], [np.float32, 0, (3)]],
            [[np.float32, 0, (5, 3)], [np.int32, 0, (5)], [np.float32, 0, (3)]],
            [[np.float32, 0, (5, 3)], [np.int32, 0, (5)], [np.int32, 0, (3)]],
            [[np.float32, 0, (5, 3)], [np.int32, 0, (5)], [np.bool_, 0, (3)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_vec1, npu_vec1 = create_common_tensor(item[1], 1, 100)
            cpu_vec2, npu_vec2 = create_common_tensor(item[2], 1, 100)
            beta = 1
            alpha = 1
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_vec1, cpu_vec2, beta, alpha)
            npu_output = self.npu_op_exec(npu_input1, npu_vec1, npu_vec2, beta, alpha)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_addr_out_common_shape_format(self):
        shape_format = [
            [
                [np.float32, 0, (5, 3)],
                [np.float32, 0, (5, 3)],
                [np.float32, 0, (5)],
                [np.float32, 0, (3)],
            ],
            [
                [np.int32, 0, (5, 3)],
                [np.int32, 0, (5, 3)],
                [np.int32, 0, (5)],
                [np.int32, 0, (3)],
            ],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 100)
            cpu_vec1, npu_vec1 = create_common_tensor(item[2], 1, 100)
            cpu_vec2, npu_vec2 = create_common_tensor(item[3], 1, 100)
            beta = 1
            alpha = 1
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_vec1, cpu_vec2, beta, alpha)
            npu_output = self.npu_op_exec_out(
                npu_input1, npu_input2, npu_vec1, npu_vec2, beta, alpha
            )
            self.assertRtolEqual(cpu_output, npu_output)

    def test_addr_out_resize(self):
        device = 'cpu'
        input_matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device=device)
        vec1 = torch.tensor([2, 3, 4], dtype=torch.float32, device=device)
        vec2 = torch.tensor([5, 6, 7], dtype=torch.float32, device=device)
        tensor_2 = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
        beta = 2
        alpha = 3
        result_cpu = torch.addr(input_matrix, vec1, vec2, beta=beta, alpha=alpha, out=tensor_2)
        result_npu = torch.addr(input_matrix.npu(), vec1.npu(), vec2.npu(), beta=beta, alpha=alpha, out=tensor_2.npu())
        self.assertRtolEqual(result_cpu, result_npu)


if __name__ == "__main__":
    run_tests()
