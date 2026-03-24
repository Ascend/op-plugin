import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestLinalgSolveTriangular(TestCase):
    def test_linalg_solve_triangular_mixed_input_dtype_with_broadcast(self):
        cpu_a = torch.tensor(
            [[[8.0, 1.0, -0.5, 2.0],
              [0.0, 7.0, 1.5, -1.0],
              [0.0, 0.0, 6.0, 0.5],
              [0.0, 0.0, 0.0, 5.0]]],
            dtype=torch.float32)
        cpu_b = torch.tensor(
            [[[1.0, -2.0, 3.0],
              [4.0, 0.5, -1.5],
              [2.0, -3.0, 1.0],
              [0.25, 1.5, -2.0]],
             [[-1.0, 2.5, 0.75],
              [3.0, -0.5, 1.25],
              [2.0, 1.0, -1.5],
              [4.0, -2.0, 0.5]]],
            dtype=torch.float16)
        expected = torch.linalg.solve_triangular(cpu_a, cpu_b, upper=True)

        npu_a = cpu_a.npu()
        npu_b = cpu_b.npu()

        self.assertEqual(npu_a.dtype, torch.float32)
        self.assertEqual(npu_b.dtype, torch.float16)
        self.assertNotEqual(npu_a.dtype, npu_b.dtype)
        actual = torch.linalg.solve_triangular(npu_a, npu_b, upper=True)

        self.assertEqual(actual.dtype, torch.float32)
        self.assertRtolEqual(expected, actual.cpu())

if __name__ == "__main__":
    run_tests()
