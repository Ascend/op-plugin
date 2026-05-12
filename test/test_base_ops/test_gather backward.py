import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestGather(TestCase):
    def cpu_op_exec(self, input1, dim, index, sparse_grad=False):
        input_tensor = input1.detach().clone().requires_grad_(True)
        output = torch.gather(input_tensor, dim, index, sparse_grad=sparse_grad)
        output.backward(torch.ones_like(output))
        output = output.detach().numpy()
        grad = input_tensor.grad
        if sparse_grad:
            self.assertTrue(grad.is_sparse)
            grad = grad.to_dense()
        return output, grad

    def npu_op_exec(self, input1, dim, index, sparse_grad=False):
        input_tensor = input1.detach().clone().requires_grad_(True)
        output = torch.gather(input_tensor, dim, index, sparse_grad=sparse_grad)
        output.backward(torch.ones_like(output))
        output = output.to("cpu")
        output = output.detach().numpy()
        grad = input_tensor.grad
        if sparse_grad:
            self.assertTrue(grad.is_sparse)
            grad = grad.to_dense().cpu()
        else:
            grad = grad.cpu()
        return output, grad

    def _run_sparse_gather_case(self, size_x, size_ind, dim):
        cpu_input = torch.randn(size_x, dtype=torch.float32, requires_grad=True)
        npu_input = cpu_input.detach().clone().to("npu").requires_grad_(True)
        if len(size_ind) > 0 and len(size_x) > 0:
            cpu_index = torch.randint(cpu_input.size(dim), size_ind, dtype=torch.int64)
        else:
            cpu_index = torch.zeros(size_ind, dtype=torch.int64)
        npu_index = cpu_index.to("npu")

        cpu_dense_output = torch.gather(cpu_input, dim, cpu_index, sparse_grad=False)
        cpu_grad = torch.rand_like(cpu_dense_output)
        cpu_dense_output.backward(cpu_grad)
        cpu_dense_grad = cpu_input.grad.clone()
        cpu_input.grad = None

        cpu_sparse_output = torch.gather(cpu_input, dim, cpu_index, sparse_grad=True)
        cpu_sparse_output.backward(cpu_grad)
        self.assertTrue(cpu_input.grad.is_sparse)
        cpu_sparse_grad = cpu_input.grad.to_dense()

        npu_grad = cpu_grad.to("npu")
        npu_sparse_output = torch.gather(npu_input, dim, npu_index, sparse_grad=True)
        npu_sparse_output.backward(npu_grad)
        self.assertTrue(npu_input.grad.is_sparse)
        npu_sparse_grad = npu_input.grad.to_dense().cpu()

        self.assertRtolEqual(cpu_dense_output.detach().numpy(), cpu_sparse_output.detach().numpy())
        self.assertRtolEqual(cpu_sparse_output.detach().numpy(), npu_sparse_output.cpu().detach().numpy())
        self.assertRtolEqual(cpu_dense_grad, cpu_sparse_grad)
        self.assertRtolEqual(cpu_dense_grad, npu_sparse_grad)

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
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2], True)
            npu_idx = item[2].to("npu")
            npu_output = self.npu_op_exec(npu_input1, item[1], npu_idx, True)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_gather_backward_with_empty_index_tensor_sparse_grad(self):
        dim = -1
        cpu_input = torch.rand([10, 5], dtype=torch.float32, requires_grad=True)
        npu_input = cpu_input.detach().clone().to("npu").requires_grad_(True)
        cpu_index = torch.randint(0, 2, [3, 0], dtype=torch.int64)
        npu_index = cpu_index.to("npu")

        cpu_output = torch.gather(cpu_input, dim, cpu_index, sparse_grad=True)
        cpu_output.sum().backward()
        self.assertTrue(cpu_input.grad.is_sparse)
        self.assertEqual(cpu_input.grad._nnz(), 0)
        cpu_grad = cpu_input.grad.to_dense()

        npu_output = torch.gather(npu_input, dim, npu_index, sparse_grad=True)
        npu_output.sum().backward()
        self.assertTrue(npu_input.grad.is_sparse)
        self.assertEqual(npu_input.grad._nnz(), 0)
        npu_grad = npu_input.grad.to_dense().cpu()

        expected_grad = torch.zeros_like(cpu_input, requires_grad=False)
        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.cpu().detach().numpy())
        self.assertRtolEqual(cpu_grad, npu_grad)
        self.assertRtolEqual(npu_grad, expected_grad)

    def test_sparse_gather_dim0(self):
        self._run_sparse_gather_case((10, 10), (5, 10), 0)

    def test_sparse_gather_dim1(self):
        self._run_sparse_gather_case((10, 10, 5), (10, 5, 5), 1)

    def test_sparse_gather_dim_neg(self):
        self._run_sparse_gather_case((10, 10, 5), (10, 10, 2), -1)

    def test_sparse_gather_ind_scalar(self):
        self._run_sparse_gather_case((10,), (), 0)

    def test_sparse_gather_x_scalar(self):
        # Covers the scalar-input branch in upstream _gather_sparse_backward.
        self._run_sparse_gather_case((), (2,), 0)

    def test_sparse_gather_both_scalar(self):
        # Covers the fully scalar edge case in upstream _gather_sparse_backward.
        self._run_sparse_gather_case((), (), 0)

    def test_sparse_gather_empty_index(self):
        self._run_sparse_gather_case((10, 5), (3, 0), -1)


if __name__ == "__main__":
    run_tests()
