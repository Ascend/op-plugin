import numpy as np
import torch_npu
import torch
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device


class TestRepeatInterleaveBackward(TestCase):

    def supported_op_exec(self, *args):
        inputs_data, repeats_data, output_grad, dim, all_back, repeat_int, repeat_tensor = args
        inputs_data = inputs_data.detach().clone().npu()
        repeats_data = repeats_data.detach().clone().npu()
        output_grad = output_grad.detach().clone().npu()
        if repeat_int:
            if all_back:
                result = torch_npu.repeat_interleave_backward_int(output_grad, inputs_data, repeats_data[0], dim)
                return result.cpu().float().detach().numpy()
            else:
                inputs_data.requires_grad_(True)
                inputs_data.retain_grad()
                if repeat_tensor:
                    y = torch.repeat_interleave(inputs_data, repeats_data[0], dim)
                else:
                    y = torch.repeat_interleave(inputs_data, int(repeats_data[0]), dim)
                y.backward(output_grad)
                return inputs_data.grad.cpu().float().detach().numpy()
        else:
            if all_back:
                result = torch_npu.repeat_interleave_backward_tensor(output_grad, inputs_data, repeats_data, dim)
                return result.cpu().float().detach().numpy()
            else:
                inputs_data.requires_grad_(True)
                inputs_data.retain_grad()
                y = torch.repeat_interleave(inputs_data, repeats_data, dim)
                y.backward(output_grad)
                return inputs_data.grad.cpu().float().detach().numpy()

    def custom_op_exec(self, *args):
        input_shape, repeat_shape, axis, data_type, repeats_type, repeat_int = args
        inputs_data = torch.rand(input_shape, dtype=data_type, requires_grad=True)
        repeat_low, repeat_high = 2, 129
        repeats_data = torch.randint(repeat_low, repeat_high, repeat_shape, dtype=repeats_type)

        if repeat_int:
            repeats = repeats_data[0]
        else:
            repeats = repeats_data
        y = torch.repeat_interleave(inputs_data.to(torch.float), repeats, axis).to(data_type)

        y_grad = torch.rand(y.shape, dtype=data_type, requires_grad=True)
        y.backward(y_grad)
        return inputs_data, repeats_data, y_grad

    def test_rms_norm(self, device="npu"):
        data_type_all = (torch.half, torch.bfloat16, torch.float)
        repeats_type_all = (
            torch.int64,
        )
        test_shape_all = (
            (40, 1, 16),
        )
        all_back = False
        if torch.__version__ == "2.0.1":
            all_back = True
        for input_shape in test_shape_all:
            axis_all = list(range(-len(input_shape), len(input_shape))) + [None]
            for axis in axis_all:
                for data_type in data_type_all:
                    for repeats_type in repeats_type_all:

                        for repeat_tensor in (False, True):
                            repeat_shape = (1,)
                            repeat_int = True
                            inputs_data, repeats_data, output_grad = self.custom_op_exec(input_shape,
                                                                                         repeat_shape,
                                                                                         axis,
                                                                                         data_type,
                                                                                         repeats_type,
                                                                                         repeat_int)
                            input_grad = inputs_data.grad.float().detach().numpy()
                            result = self.supported_op_exec(inputs_data, repeats_data, output_grad,
                                                            axis, all_back,
                                                            repeat_int, repeat_tensor)

                            self.assertRtolEqual(result, input_grad)


if __name__ == "__main__":
    run_tests()
