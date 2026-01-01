import os
import unittest
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuSaveTensor(TestCase):
    @unittest.skip("skip until CANN is updated to support aclrtLaunchHostFunc")
    def test_save_npugraph_tensor_bin(self):
        x = torch.randn([5, 5]).npu()
        torch_npu.save_npugraph_tensor(x, save_path = "./x.bin")
        
        self.assertTrue(os.path.exists("./x_device_0.bin"))
        actual = torch.load("./x_device_0.bin")
        self.assertEqual(x.dtype, actual.dtype)
        self.assertEqual(x.shape, actual.shape)
        self.assertRtolEqual(x, actual, 0.001)
    
    @unittest.skip("skip until CANN is updated to support aclrtLaunchHostFunc")
    def test_save_npugraph_tensor_pt(self):
        x = torch.randn([10, 10]).npu()
        torch_npu.save_npugraph_tensor(x, save_path = "./x.pt")
        
        self.assertTrue(os.path.exists("./x_device_0.pt"))
        actual = torch.load("./x_device_0.pt")
        self.assertEqual(x.dtype, actual.dtype)
        self.assertEqual(x.shape, actual.shape)
        self.assertRtolEqual(x, actual, 0.001)
    
    @unittest.skip("skip until CANN is updated to support aclrtLaunchHostFunc")
    def test_save_npugraph_tensor_empty(self):
        x = torch.randn([6, 0]).npu()
        torch_npu.save_npugraph_tensor(x, save_path = "./x.pt")
        
        self.assertTrue(os.path.exists("./x_device_0.pt"))
        actual = torch.load("./x_device_0.pt")
        self.assertEqual(x.dtype, actual.dtype)
        self.assertEqual(x.shape, actual.shape)


if __name__ == "__main__":
    run_tests()