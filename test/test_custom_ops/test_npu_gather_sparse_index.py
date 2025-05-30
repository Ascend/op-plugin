import math
import unittest

import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPUGatherSparseIndex(TestCase):

    def generate_input_shape(self, dtype):
        item_size = dtype.itemsize
        min_size_limit = 150 * 1024 / item_size

        H = np.random.randint(low=1, high=min_size_limit + 1)
        W = math.ceil(min_size_limit / H)
        return [H, W]

    def generate_index_shape(self, dim):
        min_size_limit = 960

        if dim == 1:
            return [961, ]

        index_shape = []
        for _ in range(dim - 1):
            shape_value = np.random.randint(low=1, high=min_size_limit + 1)
            index_shape.append(shape_value)
            min_size_limit = math.ceil(min_size_limit / shape_value)

        index_shape.append(math.ceil(min_size_limit / index_shape[-1]))
        return index_shape

    def golden_function(self, weight, index):
        num_embeddings, embedding_dim = weight.shape
        self.assertTrue(index.max() < num_embeddings, f"index should be less than inputs.shape[0], get {index.max()} and {num_embeddings}")

        embedding_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
        embedding_layer.weight.requires_grad = False
        embedding_layer.weight.data = weight

        return embedding_layer(index)

    @unittest.skip("Skip test_npu_gather_sparse_index due to low version of cann")
    @SupportedDevices(['Ascend910B'])
    def test_npu_gather_sparse_index(self):
        dim_list = [1, 2, 3, 4, 5, 6]
        dtype_list = [torch.float, torch.half, torch.bfloat16, torch.int32,
                      torch.int64, torch.int8, torch.uint8, torch.bool, torch.double]
        dim_dtype_list = [[dim, dtype]
                          for dim in dim_list
                          for dtype in dtype_list]
        for item in dim_dtype_list:
            dim = item[0]
            dtype = item[1]
            input_shape = self.generate_input_shape(dtype)
            index_shape = self.generate_index_shape(dim)
            H = input_shape[0]

            inputs_golden = torch.randn(input_shape, dtype=dtype, device="npu")
            inputs_npu = inputs_golden.clone()
            index_golden = torch.randint(0, H, index_shape).npu()
            index_npu = index_golden.clone()

            npu_out = torch_npu.npu_gather_sparse_index(inputs_npu, index_npu)
            golden_out = self.golden_function(inputs_golden, index_golden)

            self.assertEqual(npu_out, golden_out)


if __name__ == "__main__":
    run_tests()
