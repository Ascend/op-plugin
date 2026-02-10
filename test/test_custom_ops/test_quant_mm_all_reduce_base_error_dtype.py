import os
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch.distributed.distributed_c10d import ReduceOp
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU



class TestMmAllReduceBase(TestCase):
    def _test_npu_mm_all_reduce_base_error_dtype(self, input_list):
        x1, x2, pertoken_scale, dequant_scale, world_size = input_list
        hcom_name = "group_name"

        x1 = x1.npu()
        x2 = x2.npu()
        pertoken_scale = pertoken_scale.npu()
        dequant_scale = dequant_scale.npu()
        with self.assertRaisesRegex(RuntimeError, "when neither dtype of x1 or dtype of x2 is vaild"):
            out = torch_npu.npu_mm_all_reduce_base(x1, x2, hcom_name, reduce_op='sum', dequant_scale=dequant_scale,
                                                   pertoken_scale=pertoken_scale, bias=None, comm_turn=0)

    def _test_multiprocess(self, input_list):
        x1_list, x2_list, pertoken_scale, dequant_scale, world_size = input_list
        ps = []

        for i in range(world_size):
            self._test_npu_mm_all_reduce_base_error_dtype(
                [x1_list[i], x2_list[i], pertoken_scale, dequant_scale, world_size])

        for p in ps:
            p.join()

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend950'])
    def test_npu_mm_all_reduce_base(self):
        world_size = 2
        dtype = torch.float16
        x1_dtype_quant = np.int8
        x2_dtype_quant = np.float16
        dtype_dequant = np.float32
        data_format = -1
        x1_shape = [x1_dtype_quant, data_format, [256, 256]]
        x2_shape = [x2_dtype_quant, data_format, [256, 256]]
        pertoken_scale_shape = [dtype_dequant, data_format, [256]]
        dequant_scale_shape = [dtype_dequant, data_format, [256]]
        x1_list = []
        x2_list = []
        pertoken_scale, _ = create_common_tensor(pertoken_scale_shape, 1, 10) # 量化场景pertoken_scale的取值范围
        dequant_scale, _ = create_common_tensor(dequant_scale_shape, 1, 5) # 量化场景dequant_scale的取值范围
        for _ in range(world_size):
            x1, _ = create_common_tensor(x1_shape, -1, 1)
            x2, _ = create_common_tensor(x2_shape, -1, 1)
            x1_list.append(x1)
            x2_list.append(x2)
        self._test_multiprocess([x1_list, x2_list, pertoken_scale, dequant_scale, world_size])


if __name__ == '__main__':
    run_tests()
