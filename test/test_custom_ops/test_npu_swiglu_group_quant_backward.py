import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

torch.npu.config.allow_internal_format = False


class TestSwigluGroupQuantBackward(TestCase):

    def golden_swiglu_group_quant_backward(self, grad_y, x, weight=None, y_origin=None, group_index=None, clamp_limit=0.0):
        input_dtype = x.dtype
        grad_y = grad_y.float()
        x = x.float()
        if y_origin is not None:
            y_origin = y_origin.float()

        grad_weight = None
        trunc = 0

        if group_index is not None:
            for group in group_index:
                trunc += group

        if weight is not None:
            grad_weight = torch.sum(grad_y * y_origin, dim=-1, keepdim=True)
            if group_index is not None:
                original_gw_shape = grad_weight.shape
                grad_weight = grad_weight.reshape([-1, 1])
                num_rows = grad_weight.shape[0]
                row = torch.arange(num_rows, device=grad_weight.device)
                mask = (row < trunc).unsqueeze(-1).float()
                grad_weight = grad_weight * mask
                grad_weight = grad_weight.reshape(original_gw_shape)
            grad_y0 = grad_y * weight.float()
        else:
            grad_y0 = grad_y

        original_shape = x.shape
        x = x.reshape([-1, x.shape[-1]])
        H = x.shape[-1] // 2
        x0 = x[:, :H]
        x1 = x[:, H:]

        x0_truncated = x0
        x1_truncated = x1

        if clamp_limit != 0:
            x0 = torch.clamp(x0, max=clamp_limit)
            x1 = torch.clamp(x1, -clamp_limit, clamp_limit)

        sigmoid_x0 = torch.sigmoid(x0)
        silu_x0 = x0 * sigmoid_x0
        silu_grad_x0 = sigmoid_x0 * (1 + x0 * (1 - sigmoid_x0))

        grad_y0_flat = grad_y0.reshape([-1, H])
        grad_x0 = grad_y0_flat * x1 * silu_grad_x0
        grad_x1 = grad_y0_flat * silu_x0

        if clamp_limit != 0:
            mask_x0 = (x0_truncated < clamp_limit).float()
            mask_x1 = ((-clamp_limit < x1_truncated) & (x1_truncated < clamp_limit)).float()
            grad_x0 = grad_x0 * mask_x0
            grad_x1 = grad_x1 * mask_x1

        if group_index is not None:
            num_rows = grad_x0.shape[0]
            row = torch.arange(num_rows, device=grad_x0.device)
            mask = (row < trunc).unsqueeze(-1).float()
            grad_x0 = grad_x0 * mask
            grad_x1 = grad_x1 * mask

        grad_x = torch.cat([grad_x0, grad_x1], dim=-1)
        grad_x = grad_x.reshape(original_shape)
        grad_x = grad_x.to(input_dtype)
        return grad_x, grad_weight

    @SupportedDevices(['Ascend950'])
    def test_swiglu_group_quant_backward(self):
        grad_y = torch.randn([4, 8], dtype=torch.float32)
        x = torch.randn([4, 16], dtype=torch.float32)
        weight = torch.randn([4, 1], dtype=torch.float32)
        y_origin = torch.randn([4, 8], dtype=torch.float32)
        group_index = None
        clamp_limit = 5.0

        grad_x_golden, grad_weight_golden = self.golden_swiglu_group_quant_backward(grad_y,
                                                                                    x,
                                                                                    weight=weight,
                                                                                    y_origin=y_origin,
                                                                                    group_index=group_index,
                                                                                    clamp_limit=clamp_limit)

        grad_y_npu = grad_y.npu()
        x_npu = x.npu()
        weight_npu = weight.npu()
        y_origin_npu = y_origin.npu()

        grad_x_npu, grad_weight_npu = torch_npu.npu_swiglu_group_quant_backward(grad_y_npu,
                                                                                x_npu,
                                                                                weight=weight_npu,
                                                                                y_origin=y_origin_npu,
                                                                                group_index=group_index,
                                                                                clamp_limit=clamp_limit)
        grad_x_cpu = grad_x_npu.cpu()
        grad_weight_cpu = grad_weight_npu.cpu()

        self.assertRtolEqual(grad_x_golden.type(torch.float32), grad_x_cpu.type(torch.float32))
        self.assertRtolEqual(grad_weight_golden.type(torch.float32), grad_weight_cpu.type(torch.float32))


if __name__ == "__main__":
    run_tests()
