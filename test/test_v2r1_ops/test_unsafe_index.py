import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestUnsafeIndex(TestCase):

    def generate_index_data_bool(self, shape):
        cpu_input = torch.randn(shape) > 0
        npu_input = cpu_input.to("npu")
        return cpu_input, npu_input

    def cpu_op_exec(self, input1, index):
        output = torch.ops.aten._unsafe_index(input1, index)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, index):
        output = torch.ops.aten._unsafe_index(input1, index)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_unsafe_index_shape_format_tensor(self):
        # test index is tensor
        dtype_list = [np.float32, np.float16, np.int32]
        format_list = [0]
        shape_list = [[256, 10], [256, 256, 100], [5, 256, 256, 100]]
        shape_format_tensor = []
        for i in dtype_list:
            for j in format_list:
                for k in shape_list:
                    shape_format_tensor.append([[i, j, k], [np.int64, 0, (1, 2)]])   

        for item in shape_format_tensor:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_index1, npu_index1 = create_common_tensor(item[1], 1, 3)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_index1)
            npu_output = self.npu_op_exec(npu_input1, npu_index1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_unsafe_index_shape_format_tensor_tensor(self):
        # test index is [tensor, tensor]
        dtype_list = [np.float32, np.float16, np.int32]
        format_list = [0]
        shape_list = [[256, 10], [256, 256, 100], [5, 256, 256, 1000]]
        shape_format_multiTensor = []
        for i in dtype_list:
            for j in format_list:
                for k in shape_list:
                    shape_format_multiTensor.append([[i, j, k], [np.int64, 0, [1, 2]]])          

        for item in shape_format_multiTensor:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_index1, npu_index1 = create_common_tensor(item[1], 1, 3)
            cpu_index2, npu_index2 = create_common_tensor(item[1], 1, 3)
            cpu_output = self.cpu_op_exec(cpu_input1, (cpu_index1, cpu_index2))
            npu_output = self.npu_op_exec(npu_input1, (npu_index1, npu_index2))
            self.assertRtolEqual(cpu_output, npu_output)

    def test_unsafe_index_shape_format_list(self):
        # test index is list
        dtype_list = [np.float32, np.float16, np.int32]
        format_list = [0]
        shape_list = [[256, 10], [256, 256, 100], [5, 256, 256, 100]]
        shape_format_list = []
        for i in dtype_list:
            for j in format_list:
                for k in shape_list:
                    shape_format_list.append([[i, j, k], (0, 1)])        

        for item in shape_format_list:
            _, npu_input1 = create_common_tensor(item[0], 1, 100)
            with self.assertRaises(RuntimeError):
                self.npu_op_exec(npu_input1, item[1])

    def test_unsafe_index_shape_format_tensor_bool(self):
        # test index is bool tensor, which is a illegal condition
        dtype_list = [np.float32, np.float16, np.int32]
        format_list = [0]
        shape_list = [[256, 10], [256, 256, 100], [5, 256, 256, 100]]
        shape_format_tensor_bool = []
        for i in dtype_list:
            for j in format_list:
                for k in shape_list:
                    shape_format_tensor_bool.append([[i, j, k], k])

        for item in shape_format_tensor_bool:
            _, npu_input1 = create_common_tensor(item[0], 1, 100)
            _, npu_index = self.generate_index_data_bool(item[1])
            with self.assertRaises(RuntimeError):
                self.npu_op_exec(npu_input1, npu_index)

    def test_unsafe_index_shape_format_bool_x(self):
        # test index is [bool, x] , (x=1,bool,range)
        dtype_list = [np.float32, np.float16, np.int32]
        format_list = [0]
        shape_list = [[256, 10], [256, 256, 100], [5, 256, 256, 100]]
        index_list = [(True), (False), (True, 1),
                      (True, range(4)), (True, False)]
        shape_format_tensor_bool_list = [
            [[i, j, k], l] for i in dtype_list for j in format_list for k in shape_list for l in index_list
        ]       

        for item in shape_format_tensor_bool_list:
            _, npu_input1 = create_common_tensor(item[0], 1, 100)
            with self.assertRaises(RuntimeError):
                self.npu_op_exec(npu_input1, item[1])
                
    def test_unsafe_index_wrong_shape(self):
        # test index with wrong shape
        dtype_list = [np.float32, np.float16, np.int32]
        format_list = [0]
        shape_list = [[20, 10], [10, 256]]
        shape_format_tensor = []
        for i in dtype_list:
            for j in format_list:
                for k in shape_list:
                    shape_format_tensor.append([[i, j, k], [np.int64, 0, 29126]])

        for item in shape_format_tensor:
            _, npu_input1 = create_common_tensor(item[0], 1, 5)
            _, npu_index1 = create_common_tensor(item[1], 1, 5)
            with self.assertRaises(IndexError):
                self.npu_op_exec(npu_input1, npu_index1)

if __name__ == "__main__":
    run_tests()
