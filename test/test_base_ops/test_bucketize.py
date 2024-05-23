import unittest
import torch
import torch_npu
import hypothesis

from torch_npu.testing.testcase import TestCase, run_tests


class TestBucketize(TestCase):
    
    def test_bucketize(self):
        v = torch.tensor([[3, 6, 9], [3, 6, 9]])
        boundaries = torch.tensor([1, 3, 5, 7, 9])

        cpu_output = torch.bucketize(v, boundaries)
        npu_output = torch.bucketize(v.npu(), boundaries.npu())

        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
