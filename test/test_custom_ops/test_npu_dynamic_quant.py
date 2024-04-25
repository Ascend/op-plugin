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


class TestDynamicQuant(TestCase):
    def supported_op_exec(self, inputRaw, smoothScalesDummy):
        inputFp32 = inputRaw * smoothScalesDummy
        inputFp32 = inputFp32.float()
        inputAbs = inputFp32.abs()
        inputMax = inputAbs.max(dim=-1, keepdim=True)[0]
        scaleNpu = inputMax / 127
        try:
            inputScaled = inputFp32 / scaleNpu
        except ZeroDivisionError as err:
            raise err
        outputNpu = inputScaled.round()

        return [outputNpu.to(torch.int8), scaleNpu.squeeze(-1).float()]

    def custom_op_exec(self, inputNpu, smoothScalesNpu):
        return torch_npu.npu_dynamic_quant(inputNpu, smooth_scales=smoothScalesNpu)

    def generate_input(self, inputShape, dtype="float16"):
        inputDummy = np.random.random(inputShape)
        smoothScalesDummy = np.random.random(inputShape[-1])
        inputNpu = None
        if dtype == "float16":
            inputNpu = torch.from_numpy(inputDummy).to(torch.float16).npu()
            smoothScalesNpu = torch.from_numpy(smoothScalesDummy).to(torch.float16).npu()
        else:
            inputNpu = torch.from_numpy(inputDummy).to(torch.bfloat16).npu()
            smoothScalesNpu = torch.from_numpy(smoothScalesDummy).to(torch.bfloat16).npu()
        return inputNpu, smoothScalesNpu

    @SupportedDevices(['Ascend910B'])
    def test_npu_dynamic_quant(self, device="npu"):
        inputDummy, smoothScalesDummy = self.generate_input([4, 2048, 1024])

        supportedOutput = self.supported_op_exec(inputDummy.clone(), smoothScalesDummy.clone())
        customOutput = self.custom_op_exec(inputDummy.clone(), smoothScalesDummy.clone())
        self.assertTensorsSlowEqual(supportedOutput[0], customOutput[0], 1)
        self.assertRtolEqual(supportedOutput[1], customOutput[1], 0.0001)

if __name__ == "__main__":
    run_tests()