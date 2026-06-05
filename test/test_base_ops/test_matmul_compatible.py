import os
os.environ["TORCH_NPU_USE_COMPATIBLE_IMPL"] = "1"

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class TestMatMulCompatible(TestCase):
    def op_exec_cpu(self, mat1, mat2):
        input1 = mat1
        input2 = mat2
        input1.requires_grad = True
        input2.requires_grad = True

        cpu_output = torch.matmul(input1, input2)
        tmp = torch.ones_like(cpu_output)
        cpu_output.backward(tmp)

        return cpu_output.detach().numpy(), input1.grad.numpy(), input2.grad.numpy()

    def op_exec_npu(self, mat1, mat2):
        input1 = mat1
        input2 = mat2
        input1.requires_grad = True
        input2.requires_grad = True

        npu_output = torch.matmul(input1, input2)
        tmp = torch.ones_like(npu_output)
        npu_output.backward(tmp)
        npu_output = npu_output.cpu()
        return npu_output.detach().cpu().numpy(), input1.grad.cpu().numpy(), input2.grad.cpu().numpy()


    def matmul_backward_result(self, shape_format):
        for item in shape_format:
            mat1_cpu, mat1_npu = create_common_tensor(item[0], -10, 10)
            if mat1_cpu.dtype == torch.float16:
                mat1_cpu = mat1_cpu.to(torch.float32)
            mat2_cpu, mat2_npu = create_common_tensor(item[1], -10, 10)
            if mat2_cpu.dtype == torch.float16:
                mat2_cpu = mat2_cpu.to(torch.float32)
            cpu_output, cpu_mat1_grad, cpu_mat2_grad = self.op_exec_cpu(mat1_cpu, mat2_cpu)
            npu_output, npu_mat1_grad, npu_mat2_grad = self.op_exec_npu(mat1_npu, mat2_npu)

            self.assertRtolEqual(cpu_output.astype(npu_output.dtype), npu_output)
            self.assertRtolEqual(cpu_mat1_grad.astype(npu_mat1_grad.dtype), npu_mat1_grad)
            self.assertRtolEqual(cpu_mat2_grad.astype(npu_mat2_grad.dtype), npu_mat2_grad)

    def test_matmul_backward_shape_format_fp16_case1(self):
        shape_format = [
            [[np.float16, 2, [5]], [np.float16, 2, [5]]],
            [[np.float16, 2, [16]], [np.float16, 2, [16]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case3(self):
        shape_format = [
            [[np.float16, 2, [5]], [np.float16, 2, [5, 6]]],
            [[np.float16, 2, [5]], [np.float16, 2, [5, 5]]],
            [[np.float16, 2, [3, 4]], [np.float16, 2, [4]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case4(self):
        shape_format = [
            [[np.float16, 2, [5, 7]], [np.float16, 2, [7, 10]]],
            [[np.float16, 2, [5, 10]], [np.float16, 2, [10, 20]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case5(self):
        shape_format = [
            [[np.float16, 2, [4, 5, 10]], [np.float16, 2, [10]]],
            [[np.float16, 2, [5, 10, 20, 30]], [np.float16, 2, [30]]],
            [[np.float16, 2, [20, 30, 40, 50, 60]], [np.float16, 2, [60]]],
            [[np.float16, 2, [2, 3, 4, 5, 6, 8]], [np.float16, 2, [8]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case6(self):
        shape_format = [
            [[np.float16, 2, [5, 7, 10]], [np.float16, 2, [10, 16]]],
            [[np.float16, 2, [5, 10, 20, 30]], [np.float16, 2, [30, 25]]],
            [[np.float16, 2, [2, 5, 7, 8, 9, 10]], [np.float16, 2, [10, 16]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case7(self):
        shape_format = [
            [[np.float16, 2, [3, ]], [np.float16, 2, [2, 3, 2]]],
            [[np.float16, 2, [20]], [np.float16, 2, [5, 10, 20, 30]]]
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case8(self):
        shape_format = [
            [[np.float16, 2, [2, 3]], [np.float16, 2, [2, 3, 2]]],
            [[np.float16, 2, [44, 20]], [np.float16, 2, [5, 10, 20, 30]]],
            [[np.float16, 2, [75, 50]], [np.float16, 2, [2, 3, 40, 50, 60]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case9(self):
        shape_format = [
            [[np.float16, 2, [5, 7, 10]], [np.float16, 2, [5, 10, 15]]],
            [[np.float16, 2, [68, 75, 16]], [np.float16, 2, [68, 16, 43]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case10(self):
        shape_format = [
            [[np.float16, 2, [9, 1]], [np.float16, 2, [1]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case_zero_batch(self):
        shape_format = [
            [[np.float16, 2, [0, 5, 10]], [np.float16, 2, [1, 10, 16]]],
            [[np.float16, 2, [0, 8, 12]], [np.float16, 2, [1, 12, 20]]],
        ]
        self.matmul_backward_result(shape_format)

    def matmul_bf16_backward_result(self, shape_pairs):
        for mat1_shape, mat2_shape in shape_pairs:
            mat1_orig = torch.randn(mat1_shape, dtype=torch.bfloat16)
            mat2_orig = torch.randn(mat2_shape, dtype=torch.bfloat16)
            mat1_cpu = mat1_orig.float().requires_grad_(True)
            mat2_cpu = mat2_orig.float().requires_grad_(True)
            mat1_npu = mat1_orig.npu()
            mat2_npu = mat2_orig.npu()

            cpu_out = torch.matmul(mat1_cpu, mat2_cpu)
            cpu_out.backward(torch.ones_like(cpu_out))

            mat1_npu.requires_grad_(True)
            mat2_npu.requires_grad_(True)
            npu_out = torch.matmul(mat1_npu, mat2_npu)
            npu_out.backward(torch.ones_like(npu_out))

            cpu_out = cpu_out.detach().to(npu_out.dtype)
            cpu_mat1_grad = mat1_cpu.grad.to(mat1_npu.grad.dtype)
            cpu_mat2_grad = mat2_cpu.grad.to(mat2_npu.grad.dtype)

            self.assertRtolEqual(cpu_out.float().numpy(), npu_out.detach().cpu().float().numpy(), prec=0.001)
            self.assertRtolEqual(cpu_mat1_grad.float().numpy(), mat1_npu.grad.cpu().float().numpy(), prec=0.001)
            self.assertRtolEqual(cpu_mat2_grad.float().numpy(), mat2_npu.grad.cpu().float().numpy(), prec=0.001)

    def test_matmul_bf16_vec_vec(self):
        shape_pairs = [
            ([5], [5]),
            ([16], [16]),
        ]
        self.matmul_bf16_backward_result(shape_pairs)

    def test_matmul_bf16_vec_mat(self):
        shape_pairs = [
            ([5], [5, 6]),
            ([5], [5, 5]),
            ([3, 4], [4]),
        ]
        self.matmul_bf16_backward_result(shape_pairs)

    def test_matmul_bf16_mat_mat(self):
        shape_pairs = [
            ([5, 7], [7, 10]),
            ([5, 10], [10, 20]),
        ]
        self.matmul_bf16_backward_result(shape_pairs)

    def test_matmul_bf16_batch_vec(self):
        shape_pairs = [
            ([4, 5, 10], [10]),
            ([5, 10, 20, 30], [30]),
        ]
        self.matmul_bf16_backward_result(shape_pairs)

    def test_matmul_bf16_batch_mat(self):
        shape_pairs = [
            ([5, 7, 10], [10, 16]),
            ([5, 10, 20, 30], [30, 25]),
        ]
        self.matmul_bf16_backward_result(shape_pairs)

    def test_matmul_bf16_broadcast(self):
        shape_pairs = [
            ([5, 7, 10], [5, 10, 15]),
            ([68, 75, 16], [68, 16, 43]),
        ]
        self.matmul_bf16_backward_result(shape_pairs)

    def test_matmul_bf16_zero_batch(self):
        shape_pairs = [
            ([0, 5, 10], [1, 10, 16]),
            ([0, 8, 12], [1, 12, 20]),
        ]
        self.matmul_bf16_backward_result(shape_pairs)

    def test_matmul_allow_hf32(self):
        torch.npu.matmul.allow_hf32 = True
        shape_format = [
            [[np.float16, 2, [5]], [np.float16, 2, [5]]],
            [[np.float16, 2, [16]], [np.float16, 2, [16]]],
        ]
        self.matmul_backward_result(shape_format)
        torch.npu.matmul.allow_hf32 = False

if __name__ == "__main__":
    np.random.seed(1234)
    run_tests()
