import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

torch.npu.config.allow_internal_format = False


class TestSwiGluBackward(TestCase):

    def swish(self, beta, x):
        return x * torch.sigmoid(beta * x)

    def swish_backward(self, beta, x):
        return torch.sigmoid(beta * x) + x * (1 - torch.sigmoid(beta * x)) * torch.sigmoid(beta * x) * beta

    def get_golden(self, tensor_gradout, input_self_tensor, dim):

        def swiglu_backward_v1(x):
            """0.1版本，FP32格式运算，最后输出转成BF16"""
            beta_value = 1.0
            inTensors = torch.chunk(x, 2, dim=dim)
            tensor_self_float = inTensors[0].type(torch.float)
            tensor_other_float = inTensors[1].type(torch.float)
            tensor_gradout_float = tensor_gradout.type(torch.float)
            torch.mul(torch.relu(tensor_self_float), tensor_other_float)
            tensor_out1 = torch.mul(torch.mul(tensor_other_float, self.swish_backward(beta_value, tensor_self_float)),
                                    tensor_gradout_float)
            tensor_out2 = torch.mul(tensor_gradout_float, self.swish(beta_value, tensor_self_float))
            tensor_out_float = torch.cat((tensor_out1, tensor_out2), dim=-1)
            return tensor_out_float.type(torch.bfloat16)

        def swiglu_backward_v2(x):
            """Swish小算子拼接的版本"""

            beta_value = 1.0
            inTensors = torch.chunk(x, 2, dim=dim)
            tensor_self_float = inTensors[0].type(torch.float)
            tensor_other_float = inTensors[1].type(torch.float)
            tensor_gradout_float = tensor_gradout.type(torch.float)
            torch.mul(torch.relu(tensor_self_float), tensor_other_float)

            swish_grad_output = self.swish_grad(beta_value, tensor_self_float)
            swish_output = self.swish(beta_value, tensor_self_float)

            grad_out_temp1 = swish_grad_output * tensor_gradout_float
            tensor_out1 = grad_out_temp1.type(torch.bfloat16).type(torch.float) * tensor_other_float
            tensor_out2 = swish_output.type(torch.bfloat16).type(torch.float) * tensor_gradout_float
            tensor_out_float = torch.cat((tensor_out1, tensor_out2), dim=dim)
            return tensor_out_float.type(torch.bfloat16)

        def swiglu_backward_v3(x):
            """0.3版本，Silu小算子拼接的版本，最后乘法使用BF16计算"""

            x = torch.chunk(x, 2, dim=dim)
            self_tensor = x[0].npu()
            other_tensor = x[1].npu()
            tensor_gradout_npu = tensor_gradout.npu()

            self_tensor.requires_grad_(True)
            self_tensor.retain_grad()

            swish_output = F.silu(self_tensor)
            swish_output.backward([tensor_gradout_npu])
            swish_grad = self_tensor.grad

            m = swish_grad * other_tensor
            n = tensor_gradout_npu * F.silu(self_tensor)
            output = torch.cat([m, n], dim=dim)
            return output

        output = swiglu_backward_v3(input_self_tensor)
        return output

    @SupportedDevices(['Ascend910B'])
    def test_swiglu_backward(self):
        shape = [8192, 1, 3904 * 2]
        grad_shape = [8192, 1, 3904]
        dim = -1
        grad_out = torch.rand(grad_shape, device='cpu', dtype=torch.bfloat16)
        input_self_tensor = torch.rand(shape, device='cpu', dtype=torch.bfloat16)

        golden = self.get_golden(grad_out, input_self_tensor, dim)
        torch.npu.synchronize()

        input_self_tensor_npu = input_self_tensor.npu()
        input_self_tensor_npu.requires_grad_(True)
        input_self_tensor_npu.retain_grad()
        grad_out_npu = grad_out.npu()

        output_forward = torch_npu.npu_swiglu(input_self_tensor_npu, dim)
        output_forward.backward([grad_out_npu])
        result = input_self_tensor_npu.grad.cpu()
        torch.npu.synchronize()

        self.assertRtolEqual(golden.type(torch.float32), result.type(torch.float32))


if __name__ == "__main__":
    run_tests()
