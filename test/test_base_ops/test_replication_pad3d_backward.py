import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

torch.npu.set_compile_mode(jit_compile=False)


class TestReplicationPad3dBackward(TestCase):

    def replication_pad3d_backward(self, grad_out, self_tensor, padding):
        padding_layer = torch.nn.ReplicationPad3d(padding)

        self_tensor.requires_grad = True
        output = padding_layer(self_tensor)
        output.backward(grad_out)

        grad_result = self_tensor.grad
        return grad_result

    def test_replication_pad3d_backward(self):
        dtype = np.float32
        data_format = -1
        input_shape = [dtype, data_format, [1, 1, 4, 4, 4]]
        grad_shape = [dtype, data_format, [1, 1, 8, 8, 8]]
        padding = [2, 2, 2, 2, 2, 2]
        grad_out_tensor = create_common_tensor(grad_shape, -1, 1)[0]
        self_tensor = create_common_tensor(input_shape, -1, 1)[0]

        self_tensor_npu = self_tensor.clone().npu()

        golden = self.replication_pad3d_backward(grad_out_tensor, self_tensor, padding)
        output = self.replication_pad3d_backward(grad_out_tensor.npu(), self_tensor_npu, padding)

        self.assertRtolEqual(golden, output)


if __name__ == "__main__":
    run_tests()
