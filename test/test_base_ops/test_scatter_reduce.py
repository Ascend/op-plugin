import unittest
import torch
import numpy as np
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

@unittest.skip("temporarily skip scatter_reduce UT due to outdated CANN version")
class TestScatterReduce(TestCase):
    def _scatter_reduce_exec(self, input1, dim, index, src, reduce, include_self):
        if hasattr(torch, "scatter_reduce"):
            return torch.scatter_reduce(
                input1, dim, index, src, reduce=reduce, include_self=include_self
            )
        return input1.scatter_reduce(dim, index, src, reduce=reduce, include_self=include_self)

    def _scatter_reduce_exec_out(self, input1, dim, index, src, reduce, include_self, output):
        if hasattr(torch, "scatter_reduce"):
            torch.scatter_reduce(
                input1, dim, index, src, reduce=reduce, include_self=include_self, out=output
            )
            return output
        try:
            torch.ops.aten.scatter_reduce.out(input1, dim, index, src, reduce, include_self, out=output)
        except TypeError:
            torch.ops.aten.scatter_reduce.out(input1, dim, index, src, reduce, include_self, output)
        return output

    def _scatter_reduce_exec_inp(self, input1, dim, index, src, reduce, include_self):
        if hasattr(input1, "scatter_reduce_"):
            input1.scatter_reduce_(dim, index, src, reduce=reduce, include_self=include_self)
        else:
            torch.ops.aten.scatter_reduce_(input1, dim, index, src, reduce, include_self)
        return input1

    def cpu_op_exec(self, input1, dim, index, src, reduce, include_self):
        if reduce == "none":
            output = self._run_scatter_reduce_none(input1, dim, index, src, include_self)
        else:
            output = self._scatter_reduce_exec(input1, dim, index, src, reduce, include_self)
        return output.numpy()

    def npu_op_exec(self, input1, dim, index, src, reduce, include_self):
        if reduce == "none":
            output = self._run_scatter_reduce_none(input1, dim, index, src, include_self)
        else:
            output = self._scatter_reduce_exec(input1, dim, index, src, reduce, include_self)
        return output.to("cpu").numpy()

    def cpu_op_exec_out(self, input1, dim, index, src, reduce, include_self, output):
        if reduce == "none":
            self._run_scatter_reduce_none_out(input1, dim, index, src, include_self, output)
        else:
            self._scatter_reduce_exec_out(input1, dim, index, src, reduce, include_self, output)
        return output.numpy()

    def npu_op_exec_out(self, input1, dim, index, src, reduce, include_self, output):
        if reduce == "none":
            self._run_scatter_reduce_none_out(input1, dim, index, src, include_self, output)
        else:
            self._scatter_reduce_exec_out(input1, dim, index, src, reduce, include_self, output)
        return output.to("cpu").numpy()

    def cpu_op_exec_inp(self, input1, dim, index, src, reduce, include_self):
        if reduce == "none":
            self._run_scatter_reduce_none_inp(input1, dim, index, src, include_self)
        else:
            self._scatter_reduce_exec_inp(input1, dim, index, src, reduce, include_self)
        return input1.numpy()

    def npu_op_exec_inp(self, input1, dim, index, src, reduce, include_self):
        if reduce == "none":
            self._run_scatter_reduce_none_inp(input1, dim, index, src, include_self)
        else:
            self._scatter_reduce_exec_inp(input1, dim, index, src, reduce, include_self)
        return input1.to("cpu").numpy()

    def _run_scatter_reduce_none(self, input1, dim, index, src, include_self):
        try:
            return self._scatter_reduce_exec(input1, dim, index, src, "none", include_self)
        except (RuntimeError, TypeError, AttributeError):
            if input1.device.type != "cpu":
                raise
            return input1.scatter(dim, index, src)

    def _run_scatter_reduce_none_out(self, input1, dim, index, src, include_self, output):
        try:
            return self._scatter_reduce_exec_out(input1, dim, index, src, "none", include_self, output)
        except (RuntimeError, TypeError, AttributeError):
            if input1.device.type != "cpu":
                raise
            output.copy_(input1.scatter(dim, index, src))
            return output

    def _run_scatter_reduce_none_inp(self, input1, dim, index, src, include_self):
        try:
            return self._scatter_reduce_exec_inp(input1, dim, index, src, "none", include_self)
        except (RuntimeError, TypeError, AttributeError):
            if input1.device.type != "cpu":
                raise
            input1.scatter_(dim, index, src)
            return input1

    def _run_with_deterministic(self, func):
        if not hasattr(torch, "use_deterministic_algorithms"):
            return func()
        old_flag = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            return func()
        finally:
            torch.use_deterministic_algorithms(old_flag)

    def test_scatter_reduce_float32_shape_format(self):
        shape_format = [
            [0, [np.int64, 0, [10, 20]], [np.float32, 0, [10, 20]], [np.float32, 0, [10, 20]]],
            [1, [np.int64, 0, [10, 20]], [np.float32, 0, [10, 20]], [np.float32, 0, [10, 20]]],
            [0, [np.int64, 0, [2, 6]], [np.float32, 0, [2, 6]], [np.float32, 0, [2, 6]]],
            [1, [np.int64, 0, [2, 6]], [np.float32, 0, [2, 6]], [np.float32, 0, [2, 6]]],
            [0, [np.int64, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]]],
            [1, [np.int64, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]]],
            [2, [np.int64, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]]],
        ]
        reduce_list = ["none", "sum", "amin", "amax", "prod", "mean"]
        include_self_list = [True, False]

        for item in shape_format:
            for reduce in reduce_list:
                for include_self in include_self_list:
                    cpu_src, npu_src = create_common_tensor(item[2], 1, 100)
                    cpu_index, npu_index = create_common_tensor(item[1], 0, (item[1][2][item[0]] - 1))
                    cpu_input, npu_input = create_common_tensor(item[3], 1, 100)

                    cpu_output = self.cpu_op_exec(cpu_input, item[0], cpu_index, cpu_src, reduce, include_self)
                    npu_output = self.npu_op_exec(npu_input, item[0], npu_index, npu_src, reduce, include_self)
                    self.assertRtolEqual(cpu_output, npu_output)

                    cpu_out_buf = torch.empty_like(cpu_input)
                    npu_out_buf = torch.empty_like(npu_input)
                    cpu_output_out = self.cpu_op_exec_out(
                        cpu_input, item[0], cpu_index, cpu_src, reduce, include_self, cpu_out_buf
                    )
                    npu_output_out = self.npu_op_exec_out(
                        npu_input, item[0], npu_index, npu_src, reduce, include_self, npu_out_buf
                    )
                    self.assertRtolEqual(cpu_output_out, npu_output_out)

                    cpu_inp_output = self.cpu_op_exec_inp(
                        cpu_input.clone(), item[0], cpu_index, cpu_src, reduce, include_self
                    )
                    npu_inp_output = self.npu_op_exec_inp(
                        npu_input.clone(), item[0], npu_index, npu_src, reduce, include_self
                    )
                    self.assertRtolEqual(cpu_inp_output, npu_inp_output)

    def test_scatter_reduce_float16_shape_format(self):
        def cpu_op_exec_fp16(input1, dim, index, src, reduce, include_self):
            if reduce == "none":
                output = self._run_scatter_reduce_none(input1, dim, index, src, include_self)
            elif hasattr(torch, "scatter_reduce"):
                output = torch.scatter_reduce(
                    input1, dim, index, src, reduce=reduce, include_self=include_self
                )
            else:
                output = input1.scatter_reduce(dim, index, src, reduce=reduce, include_self=include_self)
            return output.float().numpy().astype(np.float16)

        def cpu_op_exec_inp_fp16(input1, dim, index, src, reduce, include_self):
            if reduce == "none":
                self._run_scatter_reduce_none_inp(input1, dim, index, src, include_self)
            elif hasattr(input1, "scatter_reduce_"):
                input1.scatter_reduce_(dim, index, src, reduce=reduce, include_self=include_self)
            else:
                torch.ops.aten.scatter_reduce_(input1, dim, index, src, reduce, include_self)
            return input1.float().numpy().astype(np.float16)

        shape_format = [
            [0, [np.int64, 0, [10, 20]], [np.float16, 0, [10, 20]], [np.float16, 0, [10, 20]]],
            [1, [np.int64, 0, [10, 20]], [np.float16, 0, [10, 20]], [np.float16, 0, [10, 20]]],
            [0, [np.int64, 0, [2, 6]], [np.float16, 0, [2, 6]], [np.float16, 0, [2, 6]]],
            [1, [np.int64, 0, [2, 6]], [np.float16, 0, [2, 6]], [np.float16, 0, [2, 6]]],
            [0, [np.int64, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]]],
            [1, [np.int64, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]]],
            [2, [np.int64, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]]],
        ]
        reduce_list = ["none", "amin", "amax"]
        include_self_list = [True, False]

        for item in shape_format:
            for reduce in reduce_list:
                for include_self in include_self_list:
                    cpu_src, npu_src = create_common_tensor(item[2], 1, 100)
                    cpu_index, npu_index = create_common_tensor(item[1], 0, (item[1][2][item[0]] - 1))
                    cpu_input, npu_input = create_common_tensor(item[3], 1, 100)

                    cpu_output = cpu_op_exec_fp16(cpu_input, item[0], cpu_index, cpu_src, reduce, include_self)
                    npu_output = self.npu_op_exec(npu_input, item[0], npu_index, npu_src, reduce, include_self)
                    self.assertRtolEqual(cpu_output, npu_output)

                    cpu_inp_output = cpu_op_exec_inp_fp16(
                        cpu_input, item[0], cpu_index, cpu_src, reduce, include_self
                    )
                    npu_inp_output = self.npu_op_exec_inp(
                        npu_input, item[0], npu_index, npu_src, reduce, include_self
                    )
                    self.assertRtolEqual(cpu_inp_output, npu_inp_output)

    def test_scatter_reduce_deterministic_case(self):
        dim = 0
        input_data = np.array(
            [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0],
             [9.0, 10.0, 11.0, 12.0],
             [13.0, 14.0, 15.0, 16.0]],
            dtype=np.float32,
        )
        index_data = np.array(
            [[0, 0, 0, 0],
             [1, 1, 1, 1],
             [0, 0, 2, 2],
             [1, 1, 2, 2]],
            dtype=np.int64,
        )
        src_data = np.array(
            [[1.0, 4.0, 3.0, 2.0],
             [5.0, 8.0, 7.0, 6.0],
             [9.0, 12.0, 11.0, 10.0],
             [13.0, 16.0, 15.0, 14.0]],
            dtype=np.float32,
        )

        reductions = ["none", "sum", "prod", "amin", "amax", "mean"]
        include_self_list = [True, False]

        def run_once(reduce, include_self):
            cpu_input = torch.tensor(input_data)
            cpu_index = torch.tensor(index_data)
            cpu_src = torch.tensor(src_data)
            npu_input = cpu_input.npu()
            npu_index = cpu_index.npu()
            npu_src = cpu_src.npu()

            if reduce == "none":
                cpu_output = self._run_scatter_reduce_none(cpu_input, dim, cpu_index, cpu_src, include_self).numpy()
                first_npu_output = self._run_scatter_reduce_none(
                    npu_input.clone(), dim, npu_index, npu_src, include_self
                ).cpu().numpy()
                second_npu_output = self._run_scatter_reduce_none(
                    npu_input.clone(), dim, npu_index, npu_src, include_self
                ).cpu().numpy()

                cpu_out_buf = torch.empty_like(cpu_input)
                npu_out_buf_first = torch.empty_like(npu_input)
                npu_out_buf_second = torch.empty_like(npu_input)
                cpu_output_out = self._run_scatter_reduce_none_out(
                    cpu_input, dim, cpu_index, cpu_src, include_self, cpu_out_buf
                ).numpy()
                first_npu_output_out = self._run_scatter_reduce_none_out(
                    npu_input.clone(), dim, npu_index, npu_src, include_self, npu_out_buf_first
                ).cpu().numpy()
                second_npu_output_out = self._run_scatter_reduce_none_out(
                    npu_input.clone(), dim, npu_index, npu_src, include_self, npu_out_buf_second
                ).cpu().numpy()

                cpu_inp_output = self._run_scatter_reduce_none_inp(
                    cpu_input.clone(), dim, cpu_index, cpu_src, include_self
                ).numpy()
                first_npu_inp_output = self._run_scatter_reduce_none_inp(
                    npu_input.clone(), dim, npu_index, npu_src, include_self
                ).cpu().numpy()
                second_npu_inp_output = self._run_scatter_reduce_none_inp(
                    npu_input.clone(), dim, npu_index, npu_src, include_self
                ).cpu().numpy()
            else:
                cpu_output = self.cpu_op_exec(cpu_input, dim, cpu_index, cpu_src, reduce, include_self)
                first_npu_output = self.npu_op_exec(npu_input.clone(), dim, npu_index, npu_src, reduce, include_self)
                second_npu_output = self.npu_op_exec(npu_input.clone(), dim, npu_index, npu_src, reduce, include_self)

                cpu_output_out = self.cpu_op_exec_out(
                    cpu_input, dim, cpu_index, cpu_src, reduce, include_self, torch.empty_like(cpu_input)
                )
                first_npu_output_out = self.npu_op_exec_out(
                    npu_input.clone(), dim, npu_index, npu_src, reduce, include_self, torch.empty_like(npu_input)
                )
                second_npu_output_out = self.npu_op_exec_out(
                    npu_input.clone(), dim, npu_index, npu_src, reduce, include_self, torch.empty_like(npu_input)
                )

                cpu_inp_output = self.cpu_op_exec_inp(cpu_input.clone(), dim, cpu_index, cpu_src, reduce, include_self)
                first_npu_inp_output = self.npu_op_exec_inp(
                    npu_input.clone(), dim, npu_index, npu_src, reduce, include_self
                )
                second_npu_inp_output = self.npu_op_exec_inp(
                    npu_input.clone(), dim, npu_index, npu_src, reduce, include_self
                )

            self.assertRtolEqual(cpu_output, first_npu_output)
            self.assertRtolEqual(first_npu_output, second_npu_output)
            self.assertRtolEqual(cpu_output_out, first_npu_output_out)
            self.assertRtolEqual(first_npu_output_out, second_npu_output_out)
            self.assertRtolEqual(cpu_inp_output, first_npu_inp_output)
            self.assertRtolEqual(first_npu_inp_output, second_npu_inp_output)

        for reduce in reductions:
            for include_self in include_self_list:
                self._run_with_deterministic(lambda r=reduce, i=include_self: run_once(r, i))


if __name__ == "__main__":
    run_tests()
