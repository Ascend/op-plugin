import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestLayernormeval(TestCase):

    def supported_op_exec(self, input1, normalized_shape):
        result = torch.nn.functional.layer_norm(input1, normalized_shape)
        return result

    def custom_op_exec(self, input1, normalized_shape):
        return torch_npu.npu_layer_norm_eval(input1, normalized_shape)

    def test_npu_layer_norm_eval(self):
        input1 = torch.rand((6, 4), dtype=torch.float32).npu()
        normalized_shape = input1.size()[1:]

        supported_result = self.supported_op_exec(input1, normalized_shape)
        custom_result = self.custom_op_exec(input1, normalized_shape)
        self.assertRtolEqual(supported_result, custom_result)


if __name__ == "__main__":
    run_tests()
