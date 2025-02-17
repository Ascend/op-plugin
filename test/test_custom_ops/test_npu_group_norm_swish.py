import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUGroupNormSwish(TestCase):

    @SupportedDevices(['Ascend910B'])
    # pylint:disable = huawei-too-many-arguments
    def supported_op_exec(self, x, weight, bias, num_groups, eps, swish_scale):
        N = x.size(0)
        C = x.size(1)
        remaining_dims = x.size()[2:]
        hw = 1
        for size in remaining_dims:
            hw *= size

        x.requires_grad_(True)
        weight.requires_grad_(True)
        bias.requires_grad_(True)

        out, mean_out, rstd_out = torch.ops.aten.native_group_norm(x, weight=weight, bias=bias, N=N, C=C, HxW=hw, group=num_groups, eps=eps)
        sigmoid_x = torch.nn.functional.sigmoid(swish_scale * out)
        out = out * sigmoid_x

        out.backward(torch.ones_like(out))

        # pylint:disable=too-many-return-values
        return out, mean_out, rstd_out, x.grad, weight.grad, bias.grad

    # pylint:disable = huawei-too-many-arguments
    def custom_op_exec(self, x, weight, bias, num_groups, data_format, eps, swish_scale):
        x.requires_grad_(True)
        weight.requires_grad_(True)
        bias.requires_grad_(True)

        out, mean_out, rstd_out = torch_npu.npu_group_norm_swish(x, num_groups, weight, bias, eps=eps, swish_scale=swish_scale)
        out.backward(torch.ones_like(out))

        # pylint:disable=too-many-return-values
        return out, mean_out, rstd_out, x.grad, weight.grad, bias.grad

    def test_npu_group_norm_swish(self):
        torch.manual_seed(123)
        shape_list = [[3, 3], [3, 6, 7, 2], [24, 35, 76]]
        dtype_list = [torch.float32]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randn(shape, dtype=dtype)
                weight = torch.randn(x.size(1), dtype=dtype)
                bias = torch.randn(x.size(1), dtype=dtype)
                x_npu = x.npu()
                weight_npu = weight.npu()
                bias_npu = bias.npu()
                eps = 1e-5
                num_groups = x.size(1)
                data_format = 'NCHW'
                swish_scale = 1.0

                cpuout = self.supported_op_exec(x, weight, bias, num_groups, eps, swish_scale)
                npuout = self.custom_op_exec(x_npu, weight_npu, bias_npu, num_groups, data_format, eps, swish_scale)

                # check forward result
                self.assertRtolEqual(cpuout[:3], npuout[:3])
                # check backward result
                self.assertRtolEqual(cpuout[3:], npuout[3:])


if __name__ == "__main__":
    run_tests()
