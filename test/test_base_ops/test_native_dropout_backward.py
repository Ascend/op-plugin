import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
import random
import numpy as np


class TestDropout(TestCase):
    @SupportedDevices(['Ascend910A', 'Ascend910B'])
    def test_native_dropout_backward_fp32(self):
        torch.manual_seed(0)
        self._test_native_dropout_backward(torch.float32, 2)

    @SupportedDevices(['Ascend910A', 'Ascend910B'])
    def test_native_dropout_backward_fp16(self):
        torch.manual_seed(0)
        self._test_native_dropout_backward(torch.float16, 3)

    @SupportedDevices(['Ascend950'])
    def test_native_dropout_backward_scale_zero(self, device="npu"):
        """A5 branch: scale == 0 (p == 1, all dropped) returns zeros."""
        torch.manual_seed(0)
        grad_output = torch.arange(32, dtype=torch.float32).reshape(32)
        mask = torch.zeros(32, dtype=torch.bool)
        output_cpu = torch.ops.aten.native_dropout_backward(grad_output, mask, 0.0)
        output_npu = torch.ops.aten.native_dropout_backward(grad_output.npu(), mask.npu(), 0.0)
        self.assertRtolEqual(output_cpu.numpy(), output_npu.cpu().numpy(), 0.0001)

    @SupportedDevices(['Ascend950'])
    def test_native_dropout_backward_scale_one(self, device="npu"):
        """A5 branch: scale == 1 (p == 0, no dropout) returns grad_output itself."""
        torch.manual_seed(0)
        grad_output = torch.arange(2 * 4 * 32, dtype=torch.float32).reshape(2, 4, 32)
        mask = torch.ones(grad_output.shape, dtype=torch.bool)
        output_cpu = torch.ops.aten.native_dropout_backward(grad_output, mask, 1.0)
        output_npu = torch.ops.aten.native_dropout_backward(grad_output.npu(), mask.npu(), 1.0)
        self.assertEqual(output_npu.shape, grad_output.shape)
        self.assertRtolEqual(output_cpu.numpy(), output_npu.cpu().numpy(), 0.0001)

    @SupportedDevices(['Ascend950'])
    def test_native_dropout_backward_scale_gt_one_fp32(self, device="npu"):
        """A5 kernel path: gradX = gradY * mask * scale (pure multiply chain, 0xAA mask pattern)."""
        torch.manual_seed(0)
        shape = (4, 32)
        grad_output = torch.arange(4 * 32, dtype=torch.float32).reshape(shape)
        packed = self._packed_bit_mask(shape, 0xAA)
        bits = self._expand_bit_mask(packed, shape)
        output_cpu = torch.ops.aten.native_dropout_backward(grad_output, bits, 2.0)
        output_npu = torch.ops.aten.native_dropout_backward(grad_output.npu(), packed.npu(), 2.0)
        self.assertRtolEqual(output_cpu.numpy(), output_npu.cpu().numpy(), 0.0001)

    @SupportedDevices(['Ascend950'])
    def test_native_dropout_backward_scale_gt_one_fp16(self, device="npu"):
        """A5 kernel path fp16: golden computed in fp32 then cast to fp16 (0x55 mask pattern)."""
        torch.manual_seed(0)
        shape = (4, 32)
        grad_output = torch.arange(4 * 32, dtype=torch.float32).reshape(shape)
        packed = self._packed_bit_mask(shape, 0x55)
        bits = self._expand_bit_mask(packed, shape)
        output_cpu = torch.ops.aten.native_dropout_backward(grad_output, bits, 3.0).to(torch.float16)
        grad_fp16 = grad_output.to(torch.float16)
        output_npu = torch.ops.aten.native_dropout_backward(grad_fp16.npu(), packed.npu(), 3.0)
        self.assertRtolEqual(output_cpu.numpy(), output_npu.cpu().numpy(), 0.001)

    @SupportedDevices(['Ascend950'])
    def test_native_dropout_backward_scale_gt_one_bf16(self, device="npu"):
        """A5 kernel path bf16: golden computed in fp32 then cast to bf16 (0x55 mask pattern)."""
        torch.manual_seed(0)
        shape = (4, 32)
        grad_output = torch.arange(4 * 32, dtype=torch.float32).reshape(shape)
        packed = self._packed_bit_mask(shape, 0x55)
        bits = self._expand_bit_mask(packed, shape)
        output_cpu = torch.ops.aten.native_dropout_backward(grad_output, bits, 3.0).to(torch.bfloat16)
        grad_bf16 = grad_output.to(torch.bfloat16)
        output_npu = torch.ops.aten.native_dropout_backward(grad_bf16.npu(), packed.npu(), 3.0)
        self.assertRtolEqual(output_cpu.float().numpy(), output_npu.cpu().float().numpy(), 0.004)

    @SupportedDevices(['Ascend950'])
    def test_native_dropout_backward_empty_mask(self, device="npu"):
        """A5 branch: empty mask returns an empty result shaped like mask.sizes()."""
        torch.manual_seed(0)
        grad_output = torch.tensor(1.2, dtype=torch.float32)
        mask = torch.zeros(0, dtype=torch.uint8)
        output_cpu = torch.ops.aten.native_dropout_backward(grad_output, mask, 2.0)
        output_npu = torch.ops.aten.native_dropout_backward(grad_output.npu(), mask.npu(), 2.0)
        self.assertRtolEqual(output_cpu.numpy(), output_npu.cpu().numpy(), 0.0001)

    @SupportedDevices(['Ascend950'])
    def test_neg_scale_range(self, device="npu"):
        """scale must be 0 or >= 1 (TORCH_CHECK on the A5 branch)."""
        torch.manual_seed(0)
        grad_output = torch.randn((4, 32), dtype=torch.float32).npu()
        mask = self._packed_bit_mask((4, 32), 0xAA).npu()
        with self.assertRaisesRegex(RuntimeError, "scale has to be 0"):
            torch.ops.aten.native_dropout_backward(grad_output, mask, 0.5)

    def _test_native_dropout_backward(self, dtype, p):
        grad_output = torch.tensor(1.2, dtype=dtype)
        b = np.random.randint(0, 100, size=(0)).astype(np.uint8)
        mask = torch.tensor(b).to(torch.uint8)
        output_cpu = torch.ops.aten.native_dropout_backward(grad_output, mask, p)
        output_npu = torch.ops.aten.native_dropout_backward(grad_output.npu(), mask.npu(), p)
        self.assertEqual(output_cpu, output_npu)

    def _packed_bit_mask(self, grad_shape, pattern):
        """Build a UINT8 bit mask with align(numel(grad), 128) / 8 elements, LSB-first per byte."""
        numel = 1
        for s in grad_shape:
            numel *= s
        packed_len = (numel + 127) // 128 * 16
        return torch.tensor([pattern] * packed_len, dtype=torch.uint8)

    def _expand_bit_mask(self, packed_mask, grad_shape):
        """Expand a packed UINT8 bit mask to a bool mask of grad_shape (LSB-first, CPU golden side)."""
        numel = 1
        for s in grad_shape:
            numel *= s
        bits = (packed_mask.unsqueeze(1) >> torch.arange(8, dtype=torch.uint8)) & 1
        return bits.bool().reshape(-1)[:numel].reshape(grad_shape)


if __name__ == '__main__':
    run_tests()
