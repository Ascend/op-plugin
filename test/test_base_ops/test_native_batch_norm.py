import unittest

import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

class TestNativeBatchNormLegit(TestCase):
    # pylint:disable = huawei-too-many-arguments
    def cpu_op_exec(self, input1, weight, bias, running_mean, running_var, training, momentum, eps):
        cpu_output = torch._native_batch_norm_legit(input1, weight, bias, running_mean, running_var, training, momentum,
            eps=eps)
        return cpu_output

    # pylint:disable = huawei-too-many-arguments
    def npu_op_exec(self, input1, weight, bias, running_mean, running_var, training, momentum, eps):
        npu_output = torch._native_batch_norm_legit(input1, weight, bias, running_mean, running_var, training, momentum,
            eps)
        return npu_output

    def test_native_batch_norm_legit(self):
        input1 = torch.randn(5, 3, 10, 10)
        weight = torch.ones(3)
        bias = torch.zeros(3)
        running_mean = torch.zeros(3)
        running_var = torch.ones(3)
        training = True
        momentum = 0.1
        eps = 1e-5
        cpu_output = self.cpu_op_exec(input1, weight, bias, running_mean, running_var, training, momentum, eps)
        npu_output = self.npu_op_exec(input1.npu(), weight.npu(), bias.npu(), running_mean.npu(), running_var.npu(), training, momentum, eps)
        self.assertRtolEqual(cpu_output[0], npu_output[0])
        self.assertRtolEqual(cpu_output[1], npu_output[1])


class TestNativeBatchNormInferSaveShapes(TestCase):
    """NPU inference save_mean/save_invstd match ATen meta ([num_features]); CPU inference uses empty ([0])."""

    def test_infer_save_tensors_match_num_features_meta_crossref(self):
        if not torch.npu.is_available():
            raise unittest.SkipTest("NPU not available")

        device = "npu"
        n, c = 3, 2
        x = torch.randn(n, c, 4, dtype=torch.float32, device=device)
        weight = torch.randn(c, dtype=torch.float32, device=device)
        bias = torch.randn(c, dtype=torch.float32, device=device)
        running_mean = torch.randn(c, dtype=torch.float32, device=device)
        running_var = torch.randn(c, dtype=torch.float32, device=device).abs().add_(1e-3)

        out_npu, sm_npu, sis_npu = torch.ops.aten.native_batch_norm.default(
            x, weight, bias, running_mean, running_var, False, -1.2, 1e-5
        )

        self.assertEqual(out_npu.shape, x.shape)
        self.assertEqual(sm_npu.shape, (c,))
        self.assertEqual(sis_npu.shape, (c,))

        x_cpu = x.cpu()
        out_cpu, sm_cpu, sis_cpu = torch.ops.aten.native_batch_norm.default(
            x_cpu,
            weight.cpu(),
            bias.cpu(),
            running_mean.cpu(),
            running_var.cpu(),
            False,
            -1.2,
            1e-5,
        )
        # CPU inference returns empty save_mean / save_invstd ([0]); NPU matches ATen meta ([num_features]).
        self.assertEqual(tuple(sm_cpu.shape), (0,))
        self.assertEqual(tuple(sis_cpu.shape), (0,))

        meta_args = (
            torch.empty(x.shape, device="meta"),
            torch.empty((c,), device="meta"),
            torch.empty((c,), device="meta"),
            torch.empty((c,), device="meta"),
            torch.empty((c,), device="meta"),
            False,
            -1.2,
            1e-5,
        )
        _, sm_meta, sis_meta = torch.ops.aten.native_batch_norm.default(*meta_args)
        self.assertEqual(tuple(sm_meta.shape), (c,))
        self.assertEqual(tuple(sis_meta.shape), (c,))


if __name__ == "__main__":
    run_tests()
