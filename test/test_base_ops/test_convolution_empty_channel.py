import torch
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

class TestConvolutionEmptyChannel(TestCase):
    channel_dim = 1

    def _build_modules(self, module_cls, module_args, module_kwargs, dtype):
        cpu_module = module_cls(*module_args, dtype=dtype, **module_kwargs)
        npu_module = module_cls(*module_args, dtype=dtype, **module_kwargs).npu()
        npu_module.load_state_dict(cpu_module.state_dict())
        return cpu_module, npu_module

    def _run_empty_case(
        self,
        module_cls,
        module_args,
        input_shape,
        dtype=torch.float32,
        module_kwargs=None):
        if module_kwargs is None:
            module_kwargs = {}
        cpu_module, npu_module = self._build_modules(module_cls, module_args, module_kwargs, dtype)

        cpu_input = torch.randn(input_shape, dtype=dtype, requires_grad=True)
        npu_input = cpu_input.detach().clone().npu().requires_grad_(True)

        cpu_output = cpu_module(cpu_input)
        npu_output = npu_module(npu_input)

        self.assertEqual(cpu_output.cpu(), npu_output.cpu())
        self.assertEqual(cpu_output.shape, npu_output.shape)
        self.assertEqual(cpu_output.numel(), 0)
        self.assertEqual(npu_output.numel(), 0)
        self.assertEqual(cpu_output.size(self.channel_dim), 0)

        cpu_grad_output = torch.rand_like(cpu_output)
        npu_grad_output = cpu_grad_output.npu()
        cpu_output.backward(cpu_grad_output)
        npu_output.backward(npu_grad_output)

        self.assertRtolEqual(cpu_input.grad, npu_input.grad.cpu())
        self.assertRtolEqual(cpu_module.weight.grad, npu_module.weight.grad.cpu())
        self.assertRtolEqual(cpu_module.bias.grad, npu_module.bias.grad.cpu())
        self.assertRtolEqual(cpu_input.grad, torch.zeros_like(cpu_input.grad))
        self.assertRtolEqual(cpu_module.weight.grad, torch.zeros_like(cpu_module.weight.grad))
        self.assertRtolEqual(cpu_module.bias.grad, torch.zeros_like(cpu_module.bias.grad))

    @SupportedDevices(['Ascend910B'])
    def test_conv_empty_channel_fp32(self):
        test_cases = [
            (nn.Conv1d, (0, 8, 2), (2, 0, 15), {"stride": 2}),
            (nn.Conv2d, (0, 33, 3), (2, 0, 50, 100), {"stride": 2}),
            (nn.Conv3d, (0, 33, 3), (2, 0, 50, 20, 40), {"stride": 2}),
        ]

        for module_cls, module_args, input_shape, module_kwargs in test_cases:
            with self.subTest(module=module_cls.__name__, input_shape=input_shape):
                self._run_empty_case(
                    module_cls,
                    module_args,
                    input_shape,
                    module_kwargs=module_kwargs)


if __name__ == "__main__":
    run_tests()
