import os
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestElewiseQuantOperation(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        os.environ["ASCEND_LAUNCH_BLOCKING"] = "0"

    def calculate_benchmark(self, input_x, input_scale, input_offset):
        input_x_np = input_x.cpu().numpy()
        input_scale_np = input_scale.cpu().numpy()
        input_offset_np = input_offset.cpu().numpy()

        if len(input_offset_np) == 0:
            out_np = np.clip((np.round((input_x_np / input_scale_np))), -128, 127)
        else:
            out_np = np.clip((np.round((input_x_np / input_scale_np)) + input_offset_np), -128, 127)
        out = torch.from_numpy(out_np).to(torch.int8)
        return out

    @SupportedDevices(['Ascend910B'])
    def test_quantpertensor(self):
        torch.manual_seed(6)
        input_x = torch.rand(1, 16, 16, dtype=torch.float16)
        input_scale = torch.rand(16, 16, dtype=torch.float16)
        input_offset = torch.randint(-10, 10, size=(16, 16), dtype=torch.int8)
        y = torch.zeros(size=(1, 16, 16), dtype=torch.int8).npu()
        out = self.calculate_benchmark(input_x, input_scale, input_offset)

        input_x = input_x.npu()
        input_scale = input_scale.npu()
        input_offset = input_offset.npu()
        torch_npu._npu_quantize_per_tensor(input_x, input_scale, input_offset, y)
        self.assertEqual(out, y)

    @SupportedDevices(['Ascend910B'])
    def test_quantpertensor_aclgraph(self):
        input_x = torch.ones(1, 16, 16, dtype=torch.float16).npu()
        input_scale = torch.ones(16, 16, dtype=torch.float16).npu()
        input_offset = torch.ones(16, 16, dtype=torch.int8).npu()
        y = torch.zeros(size=(1, 16, 16), dtype=torch.int8).npu()
        y1 = torch.zeros(size=(1, 16, 16), dtype=torch.int8).npu()

        torch_npu._npu_quantize_per_tensor(input_x, input_scale, input_offset, y)

        s = torch_npu.npu.Stream()
        with torch_npu.npu.stream(s):
            g = torch_npu.npu.NPUGraph()
            g.capture_begin()
            torch_npu._npu_quantize_per_tensor(input_x, input_scale, input_offset, y1)
            g.capture_end()

        g.replay()
        self.assertEqual(y1, y)


if __name__ == "__main__":
    run_tests()
