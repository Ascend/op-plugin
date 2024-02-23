import math
import unittest
import copy
import struct
from struct import pack, unpack
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestQuantScatter(TestCase):
    def supported_op_exec(self, var, indices, updates, quant_scales):
        quant_scales_new = quant_scales.view(32)
        updates_new = torch_npu.npu_quantize(updates, quant_scales_new, None, torch.qint8, -1).to(torch.int8)
        return torch_npu.scatter_update(var, indices, updates_new, -2)

    def custom_op_exec(self, var, indices, updates, quant_scales):
        return torch_npu.npu_quant_scatter(var, indices, updates, quant_scales, None, -2, -1, "update")

    @SupportedDevices(['Ascend910B'])
    def test_npu_quant_scatter(self, device="npu"):
        var_data = np.random.uniform(0, 1, [1, 1, 32]).astype(np.int8)
        var1 = torch.from_numpy(var_data).to(torch.int8).npu()
        var2 = var1.clone()

        indices_data = np.random.uniform(0, 1, [1]).astype(np.int32)
        indices1 = torch.from_numpy(indices_data).to(torch.int32).npu()
        indices2 = indices1.clone()

        updates_data = np.random.uniform(1, 2, [1, 1, 32]).astype(np.float16)
        updates1 = torch.from_numpy(updates_data).to(torch.bfloat16).npu()
        updates2 = updates1.clone()

        quant_scales_data = np.random.uniform(0, 1, [1, 1, 32]).astype(np.float16)
        quant_scales1 = torch.from_numpy(quant_scales_data).to(torch.bfloat16).npu()
        quant_scales2 = quant_scales1.clone()

        supported_output = self.supported_op_exec(var1, indices1, updates1, quant_scales1)
        custom_output = self.custom_op_exec(var2, indices2, updates2, quant_scales2)
        self.assertRtolEqual(supported_output, custom_output, 0.001)

if __name__ == "__main__":
    run_tests()