import torch
import torch_npu
import math

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestIndexing(TestCase):
    def setUp(self):
        super().setUp()
        self._cpu_input, self._npu_input = self.create_tensor((2, 3, 4, 5))

    @staticmethod
    def create_tensor(shape, dtype=torch.float32):
        tensor = torch.arange(math.prod(shape), dtype=dtype).reshape(shape)
        return tensor.numpy(), tensor.npu()

    def cpu_op_exec(self, input_tensor, begin, end, strides):
        slices = tuple(slice(b, e, s) for b, e, s in zip(begin, end, strides))
        return input_tensor[slices]

    def npu_op_exec(
        self,
        input_tensor,
        begin,
        end,
        strides,
        begin_mask=0,
        end_mask=0,
        ellipsis_mask=0,
        new_axis_mask=0,
        shrink_axis_mask=0,
        out=None,
    ):
        if out is not None:
            return (
                torch_npu.npu_indexing.out(
                    input_tensor,
                    begin,
                    end,
                    strides,
                    begin_mask,
                    end_mask,
                    ellipsis_mask,
                    new_axis_mask,
                    shrink_axis_mask,
                    out=out,
                )
                .cpu()
                .numpy()
            )
        return (
            torch_npu.npu_indexing(
                input_tensor, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask
            )
            .cpu()
            .numpy()
        )

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_0_all_demensions(self):
        """覆盖各种维度"""
        data_list = [
            ([9], [1], [8], [2]),
            ([6, 7], [1, 2], [4, 6], [3, 1]),
            ([10, 11, 12], [3, 2, 1], [10, 11, 12], [1, 3, 5]),
            ([2, 3, 4, 5, 6], [1, 0, 2, 1, 1], [2, 3, 4, 4, 5], [2, 1, 7, 3, 1]),
            (
                [2, 2, 2, 2, 2, 2, 2, 2],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 2, 2, 2, 2, 2],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ),
        ]
        dtype_list = [torch.int8, torch.float16]
        for shape, begin, end, strides in data_list:
            for dtype in dtype_list:
                cpu_input, npu_input = self.create_tensor(shape, dtype)
                cpu_output = self.cpu_op_exec(cpu_input, begin, end, strides)
                npu_output = self.npu_op_exec(npu_input, begin, end, strides)
                self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_1_basic_slicing_all_dimensions(self):
        """基础切片测试"""
        # 测试每个维度的基础切片
        cpu_output = self._cpu_input[0:2, 1:3, 2:4, 3:5]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1, 2, 3], [2, 3, 4, 5], [1, 1, 1, 1])
        self.assertRtolEqual(cpu_output, npu_output)
        # out
        npu_output = torch.zeros(16, dtype=self._npu_input.dtype).reshape(2, 2, 2, 2).npu()
        self.npu_op_exec(self._npu_input, [0, 1, 2, 3], [2, 3, 4, 5], [1, 1, 1, 1], out=npu_output)
        self.assertRtolEqual(cpu_output, npu_output.cpu().numpy())

        # 使用负索引
        cpu_output = self._cpu_input[0:-1, 1:-1, -3:-1, -2:5:1]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1, -3, -2], [-1, -1, -1, 5], [1, 1, 1, 1])
        self.assertRtolEqual(cpu_output, npu_output)

        # 切片维度不足输入维度
        cpu_output = self._cpu_input[0:2, 1:3]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1], [2, 3], [1, 1])
        self.assertRtolEqual(cpu_output, npu_output)

        # 切片维度为空
        cpu_output = self._cpu_input[...]
        npu_output = self.npu_op_exec(self._npu_input, [], [], [])
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_2_strides_all_directions(self):
        """测试各种步长组合"""
        # 正向步长
        cpu_output = self._cpu_input[0:2:2, 0:3:1, 0:4:2, 0:5:3]
        npu_output = self.npu_op_exec(self._npu_input, [0, 0, 0, 0], [2, 3, 4, 5], [2, 1, 2, 3])
        self.assertRtolEqual(cpu_output, npu_output)

        # 负向步长
        cpu_output = self._cpu_input[1:0:-1, 2:0:-2, 3:0:-1, 4:0:-3]
        npu_output = self.npu_op_exec(self._npu_input, [1, 2, 3, 4], [0, 0, 0, 0], [-1, -2, -1, -3])
        self.assertRtolEqual(cpu_output, npu_output)

        # 混合方向步长
        cpu_output = self._cpu_input[0:2:1, 3:0:-1, 0:4:2, 5:0:-2]
        npu_output = self.npu_op_exec(self._npu_input, [0, 3, 0, 5], [2, 0, 4, 0], [1, -1, 2, -2])
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_3_begin_mask_all_combinations(self):
        """测试所有begin_mask组合"""
        # mask第一维
        cpu_output = self._cpu_input[:2, 1:3, 2:4, 3:5]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1, 2, 3], [2, 3, 4, 5], [1, 1, 1, 1], begin_mask=0b0001)
        self.assertRtolEqual(cpu_output, npu_output)

        # mask第二维
        cpu_output = self._cpu_input[0:2, :3, 2:4, 3:5]
        npu_output = self.npu_op_exec(self._npu_input, [0, 0, 2, 3], [2, 3, 4, 5], [1, 1, 1, 1], begin_mask=0b0010)
        self.assertRtolEqual(cpu_output, npu_output)

        # mask所有维度
        cpu_output = self._cpu_input[:2, :3, :4, :5]
        npu_output = self.npu_op_exec(self._npu_input, [0, 0, 0, 0], [2, 3, 4, 5], [1, 1, 1, 1], begin_mask=0b1111)
        self.assertRtolEqual(cpu_output, npu_output)

        # mask部分维度
        cpu_output = self._cpu_input[:2, 1:, 0:, :5]
        npu_output = self.npu_op_exec(
            self._npu_input,
            [0, 1, 0, 2],
            [2, 3, 3, 5],
            [1, 1, 1, 1],
            begin_mask=0b1001,
            end_mask=0b0110,
        )
        self.assertRtolEqual(cpu_output, npu_output)

        # 忽略超出的维度
        cpu_output = self._cpu_input[0:2, 1:3]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1], [2, 3], [1, 1], begin_mask=0b1000)
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_4_end_mask_all_combinations(self):
        """测试所有end_mask组合"""
        # mask第一维
        cpu_output = self._cpu_input[0:, 1:3, 2:4, 3:5]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1, 2, 3], [0, 3, 4, 5], [1, 1, 1, 1], end_mask=0b0001)
        self.assertRtolEqual(cpu_output, npu_output)

        # mask多个维度
        cpu_output = self._cpu_input[0:, 1:, 2:, 3:]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1, 2, 3], [0, 0, 0, 0], [1, 1, 1, 1], end_mask=0b1111)
        self.assertRtolEqual(cpu_output, npu_output)

        # 忽略超出的维度
        cpu_output = self._cpu_input[0:2, 1:3]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1], [2, 3], [1, 1], end_mask=0b1000)
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_5_shrink_axis_mask_all_combinations(self):
        """测试所有shrink_axis_mask组合"""
        # 缩减单个维度
        cpu_output = self._cpu_input[0, 1:3, 0:2, 1:5:2]
        npu_output = self.npu_op_exec(
            self._npu_input, [0, 1, 0, 1], [1, 3, 2, 5], [1, 1, 1, 2], shrink_axis_mask=0b0001
        )
        self.assertRtolEqual(cpu_output, npu_output)

        # 缩减两个维度
        cpu_output = self._cpu_input[0, 1, 0:3:2, 1:5]
        npu_output = self.npu_op_exec(
            self._npu_input, [0, 1, 0, 1], [1, 2, 3, 5], [3, 2, 2, 1], shrink_axis_mask=0b0011
        )
        self.assertRtolEqual(cpu_output, npu_output)

        # 缩减所有维度
        cpu_output = self._cpu_input[0, 1, 2, 3]
        npu_output = self.npu_op_exec(
            self._npu_input, [0, 1, 2, 3], [1, 2, 3, 4], [1, 1, 1, 1], shrink_axis_mask=0b1111
        )
        self.assertEqual(cpu_output.item(), npu_output.item())

        # 缩减中间维度
        cpu_output = self._cpu_input[1:2, 1, 0:3, 2]
        npu_output = self.npu_op_exec(
            self._npu_input, [1, 1, 0, 2], [2, 2, 3, 3], [1, 1, 1, 1], shrink_axis_mask=0b1010
        )
        self.assertRtolEqual(cpu_output, npu_output)

        # 忽略超出的维度
        cpu_output = self._cpu_input[0:2, 1:3]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1], [2, 3], [1, 1], shrink_axis_mask=0b1000)
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_6_new_axis_mask_all_combinations(self):
        """测试所有new_axis_mask组合"""
        # 在开头新增维度
        cpu_output = self._cpu_input[None, 0:2, 1:3, 1:4, 2:5]
        npu_output = self.npu_op_exec(
            self._npu_input, [0, 0, 1, 1, 2], [0, 2, 3, 4, 5], [1, 1, 1, 1, 1], new_axis_mask=0b000001
        )
        self.assertRtolEqual(cpu_output, npu_output)

        # 在中间新增维度
        cpu_output = self._cpu_input[1:2, None, 1:3, None, 2:4, 1:5]
        npu_output = self.npu_op_exec(
            self._npu_input, [1, 0, 1, 0, 2, 1], [2, 0, 3, 0, 4, 5], [1, 1, 1, 1, 1, 1], new_axis_mask=0b001010
        )
        self.assertRtolEqual(cpu_output, npu_output)

        # 在所有位置新增维度
        cpu_output = self._cpu_input[None, None, None, None, 1:2, 2:3, 1:4, 2:5]
        npu_output = self.npu_op_exec(
            self._npu_input,
            [0, 0, 0, 0, 1, 2, 1, 2],
            [0, 0, 0, 0, 2, 3, 4, 5],
            [1, 1, 1, 1, 1, 1, 1, 1],
            new_axis_mask=0b00001111,
        )
        self.assertRtolEqual(cpu_output, npu_output)

        # 忽略超出的维度
        cpu_output = self._cpu_input[0:2, 1:3]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1], [2, 3], [1, 1], new_axis_mask=0b1000)
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_7_ellipsis_mask_various_positions(self):
        """测试省略号在不同位置"""
        # 省略号在中间
        cpu_output = self._cpu_input[0:1, 0:3, ..., 2:3]
        npu_output = self.npu_op_exec(self._npu_input, [0, 0, 0, 2], [1, 3, 4, 3], [1, 1, 1, 1], ellipsis_mask=0b0100)
        self.assertRtolEqual(cpu_output, npu_output)

        # 省略号在开头
        cpu_output = self._cpu_input[..., 0:3, 1:2, 2:3]
        npu_output = self.npu_op_exec(self._npu_input, [0, 0, 1, 2], [2, 3, 2, 3], [1, 1, 1, 1], ellipsis_mask=0b0001)
        self.assertRtolEqual(cpu_output, npu_output)

        # 省略号在结尾
        cpu_output = self._cpu_input[0:1, 1:2, 0:4, ...]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1, 0, 0], [1, 2, 4, 5], [1, 1, 1, 1], ellipsis_mask=0b1000)
        self.assertRtolEqual(cpu_output, npu_output)

        # 省略号不生效
        cpu_output = self._cpu_input[0:2, ..., 2:3, 1:3, 0:5:2]
        npu_output = self.npu_op_exec(
            self._npu_input,
            [0, 0, 2, 1, 0],
            [2, 3, 3, 3, 5],
            [1, 1, 1, 1, 2],
            ellipsis_mask=0b00010,
        )  # 在第2维使用省略号
        self.assertRtolEqual(cpu_output, npu_output)

        # 忽略超出的维度
        cpu_output = self._cpu_input[0:2, 1:3]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1], [2, 3], [1, 1], ellipsis_mask=0b1000)
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_8_complex_combinations_all_masks(self):
        """测试复杂的mask组合"""
        # 组合begin_mask, end_mask, strides
        cpu_output = self._cpu_input[:2, 1:3:2, 4::-1, 2:5]
        npu_output = self.npu_op_exec(
            self._npu_input,
            begin=[1, 1, 4, 2],
            end=[2, 3, -1, 5],
            strides=[1, 2, -1, 1],
            begin_mask=0b0001,  # mask第一维
            end_mask=0b0100,  # mask第三维
        )
        self.assertRtolEqual(cpu_output, npu_output)

        # 组合shrink_axis_mask和new_axis_mask
        cpu_output = self._cpu_input[0, None, None, 1, 0:4, 2:3]
        npu_output = self.npu_op_exec(
            self._npu_input,
            begin=[0, 0, 0, 1, 0, 2],
            end=[1, 0, 0, 2, 4, 3],
            strides=[1, 1, 1, 1, 1, 1],
            new_axis_mask=0b000110,  # 在第二维位置新增2个维度
            shrink_axis_mask=0b001001,  # 缩减第一维和第四维
        )
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_9_edge_cases_with_all_masks(self):
        """边界情况测试"""
        # 空切片
        cpu_output = self._cpu_input[0:0, 1:1, 2:2, 3:3]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1, 2, 3], [0, 1, 2, 3], [1, 1, 1, 1])
        self.assertRtolEqual(cpu_output, npu_output)

        # 全部使用负索引
        cpu_output = self._cpu_input[-2:2:1, -3:3:1, -4:4:1, -5:5:1]
        npu_output = self.npu_op_exec(self._npu_input, [-2, -3, -4, -5], [2, 3, 4, 5], [1, 1, 1, 1])
        self.assertRtolEqual(cpu_output, npu_output)

        # 混合mask和负索引
        cpu_output = self._cpu_input[:, -2:3:1, :, -1:5:1]
        npu_output = self.npu_op_exec(
            self._npu_input,
            begin=[0, -2, 0, -1],
            end=[2, 3, 4, 5],
            strides=[1, 1, 1, 1],
            begin_mask=0b0101,
            end_mask=0b0101,
        )
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_10_ellipsis_with_other_masks(self):
        """测试省略号与其他mask的组合"""
        # 省略号与begin/end mask组合
        cpu_output = self._cpu_input[..., 1:, :2]
        npu_output = self.npu_op_exec(
            self._npu_input,
            begin=[0, 1, 0],
            end=[2, 3, 2],
            strides=[1, 1, 1],
            ellipsis_mask=0b0001,  # 在前两维使用省略号
            begin_mask=0b100,  # mask第四维的begin
            end_mask=0b010,  # mask第三维的end
        )
        self.assertRtolEqual(cpu_output, npu_output)

        # 省略号与负步长
        cpu_output = self._cpu_input[..., ::-1, :]
        npu_output = self.npu_op_exec(
            self._npu_input,
            begin=[0, 0, 4, 0],
            end=[2, 3, 0, 5],
            strides=[1, 1, -1, 1],
            ellipsis_mask=0b1000,
            end_mask=0b0100,
        )
        self.assertRtolEqual(cpu_output, npu_output)

        # ellipsis_mask与shrink_axis_mask和new_axis_mask在同一维同时设置，ellipsis_mask优先
        cpu_output = self._cpu_input[..., 1:3, 0:2]
        npu_output = self.npu_op_exec(
            self._npu_input,
            begin=[0, 1, 0],
            end=[2, 3, 2],
            strides=[1, 1, 1],
            ellipsis_mask=0b001,  # 在前两维使用省略号
            new_axis_mask=0b001,
            shrink_axis_mask=0b001,
        )
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_11_exception_scenarios(self):
        """测试异常场景"""
        # 切片维度超出输入维度
        with self.assertRaises(Exception):
            self.npu_op_exec(self._npu_input, [0, 1, 2, 3, 0], [2, 3, 4, 5, 6], [1, 1, 1, 1, 1])

        # 步长为0的情况
        with self.assertRaises(Exception):
            self.npu_op_exec(self._npu_input, [0, 0, 0, 0], [2, 3, 4, 5], [0, 1, 1, 1])

        # shrink_axis_mask 对应的步长为负
        with self.assertRaises(Exception):
            self.npu_op_exec(self._npu_input, [0, 1, 0, 0], [1, 2, 4, 5], [-1, 1, 1, 1], shrink_axis_mask=0b0011)

        # begin/end/strides长度不匹配
        with self.assertRaises(Exception):
            self.npu_op_exec(self._npu_input, [0, 0, 0], [2, 3, 4, 5], [1, 1, 1, 1])  # begin长度不足
        with self.assertRaises(Exception):
            self.npu_op_exec(self._npu_input, [0, 0, 0, 0], [2, 3, 4], [1, 1, 1, 1])  # end长度不足
        with self.assertRaises(Exception):
            self.npu_op_exec(self._npu_input, [0, 0, 0, 0], [2, 3, 4, 5], [1, 1, 1])  # strides长度不足

        # 多个ellipsis_mask位被设置（多个省略号）
        with self.assertRaises(Exception):
            self.npu_op_exec(
                self._npu_input, [0, 0, 0, 0], [2, 3, 4, 5], [1, 1, 1, 1], ellipsis_mask=0b1010
            )  # 两个省略号

        # output shape 维度>8
        with self.assertRaises(Exception):
            self.npu_op_exec(
                self._npu_input,
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 2, 3, 4, 5],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                new_axis_mask=0b11111,
            )

        # 步长与索引方向矛盾时的边界情况
        result = self.npu_op_exec(self._npu_input, [1, 0, 0, 0], [0, 3, 4, 5], [1, 1, 1, 1])
        self.assertEqual(result.size, 0)

        # 索引越界但未超出范围太多（应该被clamp，而不是抛出异常）
        result = self.npu_op_exec(self._npu_input, [-10, -10, -10, -10], [10, 10, 10, 10], [1, 1, 1, 1])
        expected = self._cpu_input[:, :, :, :]  # 应该相当于全切片
        self.assertRtolEqual(result, expected)

        # 测试begin=0, end=0, stride=-1
        result = self.npu_op_exec(self._npu_input, [0, 0, 0, 0], [0, 0, 0, 0], [-1, -1, -1, -1])
        self.assertEqual(result.size, 0)


if __name__ == "__main__":
    run_tests()
