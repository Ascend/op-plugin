import math
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestSwigluBackwardMxQuantWithDualAxis(TestCase):
    def npu_op_exec(self, x, y_grad, group_index=None, activate_left=True, round_mode="rint",
                    scale_alg=1, dst_type=torch_npu.float8_e4m3fn, dst_type_max=0):
        return torch_npu._npu_swiglu_backward_mx_quant_with_dual_axis(x, y_grad,
                                                                      group_index=group_index,
                                                                      activate_left=activate_left,
                                                                      round_mode=round_mode,
                                                                      scale_alg=scale_alg,
                                                                      dst_type=dst_type,
                                                                      dst_type_max=dst_type_max)

    def generate_input(self, shape, value, dtype="bfloat16"):
        if dtype == "float16":
            data_type = torch.float16
        elif dtype == "bfloat16":
            data_type = torch.bfloat16
        return torch.full(shape, value, dtype=data_type)

    def _expected_mxscale1_shape(self, x_shape):
        shape = list(x_shape)
        shape[-1] = math.ceil(shape[-1] / 64)
        shape.append(2)
        return shape

    def _expected_mxscale2_shape(self, x_shape, group_index=None):
        shape = list(x_shape)
        m = shape[-2]
        if group_index is not None:
            quant_size = math.floor(m / 64) + group_index.shape[0]
        else:
            quant_size = math.ceil(m / 64)
        shape[-2] = quant_size
        shape.append(2)
        return shape

    @SupportedDevices(['Ascend950'])
    def test_npu_swiglu_backward_mx_quant_with_dual_axis_output_shape(self, device="npu"):
        # x last_dim=128, 64对齐; y_grad last_dim = x last_dim // 2 = 64
        x = self.generate_input([128, 128], value=1.0, dtype="bfloat16").to(device)
        y_grad = self.generate_input([128, 64], value=1.0, dtype="bfloat16").to(device)
        group_index = torch.tensor([2, 4], dtype=torch.int64).npu()

        y1_out, mxscale1, y2_out, mxscale2 = self.npu_op_exec(x, y_grad, group_index)

        # y1_out / y2_out 与 x 同形
        self.assertEqual(list(y1_out.shape), [128, 128])
        self.assertEqual(list(y2_out.shape), [128, 128])
        # mxscale1: [128, ceil(128/64), 2] = [128, 2, 2]
        self.assertEqual(list(mxscale1.shape), self._expected_mxscale1_shape([128, 128]))
        # mxscale2: floor(128/64)+2=4 → [4, 128, 2]
        self.assertEqual(list(mxscale2.shape), self._expected_mxscale2_shape([128, 128], group_index.cpu()))

    @SupportedDevices(['Ascend950'])
    def test_npu_swiglu_backward_mx_quant_with_dual_axis_without_group_index(self, device="npu"):
        # 无 group_index 时 mxscale2 用 ceil 计算 quant_size
        x = self.generate_input([128, 128], value=1.0, dtype="bfloat16").to(device)
        y_grad = self.generate_input([128, 64], value=1.0, dtype="bfloat16").to(device)

        y1_out, mxscale1, y2_out, mxscale2 = self.npu_op_exec(x, y_grad)

        self.assertEqual(list(y1_out.shape), [128, 128])
        self.assertEqual(list(y2_out.shape), [128, 128])
        self.assertEqual(list(mxscale1.shape), self._expected_mxscale1_shape([128, 128]))
        # mxscale2: ceil(128/64)=2 → [2, 128, 2]
        self.assertEqual(list(mxscale2.shape), self._expected_mxscale2_shape([128, 128]))

    def golden_op_exec_backward(self, input_tensor):
        # x=[4,64]=1.0 bf16, y_grad=[4,32]=1.0 bf16, group_index=[2,4], activate_left=True
        row_y = [240] * 32 + [128] * 32
        y1_out = torch.tensor([row_y] * 4, dtype=torch.uint8)
        mxscale1 = torch.tensor([[[119, 118]]] * 4, dtype=torch.uint8)
        y2_out = torch.tensor([row_y] * 4, dtype=torch.uint8)
        col_scale = [[119, 0]] * 32 + [[118, 0]] * 32
        mxscale2 = torch.tensor([col_scale] * 2, dtype=torch.uint8)
        return y1_out, mxscale1, y2_out, mxscale2

    @SupportedDevices(['Ascend950'])
    def test_npu_swiglu_backward_mx_quant_with_dual_axis_precision(self, device="npu"):
        x = self.generate_input([4, 64], value=1.0, dtype="bfloat16")
        x = x.to(device).requires_grad_(True)
        y_grad = self.generate_input([4, 32], value=1.0, dtype="bfloat16").to(device)
        group_index = torch.tensor([2, 4], dtype=torch.int64).npu()

        golden_output = self.golden_op_exec_backward(x.clone().detach())
        npu_output = self.npu_op_exec(x, y_grad, group_index)

        y1_out = npu_output[0].cpu().view([4, 64]).to(torch.uint8)
        mxscale1 = npu_output[1].cpu().view([4, 1, 2]).to(torch.uint8)
        y2_out = npu_output[2].cpu().view([4, 64]).to(torch.uint8)
        mxscale2 = npu_output[3].cpu().view([2, 64, 2]).to(torch.uint8)

        assert torch.all(y1_out == golden_output[0].view(torch.uint8))
        assert torch.all(mxscale1 == golden_output[1].view(torch.uint8))
        assert torch.all(y2_out == golden_output[2].view(torch.uint8))
        assert torch.all(mxscale2 == golden_output[3].view(torch.uint8))


if __name__ == "__main__":
    run_tests()
