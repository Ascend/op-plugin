import unittest
import itertools
import importlib.util
from os import scandir
import numpy as np
import random
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


def numpy_to_torch(np_arr):
    FP8_DTYPE_MAP_NUMPY_TO_TORCH = {
        "bfloat16": torch.bfloat16,
        "float8_e5m2": torch.float8_e5m2,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e8m0": None if not hasattr(torch, "float8_e8m0fnu") else getattr(torch, "float8_e8m0fnu")
    }

    def _bitcast_float8_to_torch(np_arr):
        np_dtype = np_arr.dtype
        torch_dtype = FP8_DTYPE_MAP_NUMPY_TO_TORCH[str(np_dtype)]
        np_uint8 = np_arr.view(np.uint8)
        t_uint8 = torch.from_numpy(np_uint8)
        return t_uint8.view(torch_dtype)
    
    if str(np_arr.dtype) in list(FP8_DTYPE_MAP_NUMPY_TO_TORCH.keys()):
        return _bitcast_float8_to_torch(np_arr)
    return torch.from_numpy(np_arr)


def assert_tensors_close(x: torch.Tensor, y: torch.Tensor, rtol=1e-3, atol=1e-5, label="Tensor"):
    # 1. 确保在 CPU
    if x.device.type != "cpu":
        x = x.cpu()
    if y.device.type != "cpu":
        y = y.cpu()

    # 2. 统一转为 float32 进行高精度比对
    #    (无论是 int 还是 float8，转为 float32 都能保证计算精度)
    x_f32 = x.to(torch.float32)
    y_f32 = y.to(torch.float32)

    # 3. 检查 NaN 和 Inf (FP8 极易产生这些)
    #    如果两边都是 NaN，通常视为相等；如果一个是 NaN 一个不是，则不等。
    #    torch.equal 处理 NaN 比较严格，这里我们手动检查
    nan_mask_x = torch.isnan(x_f32)
    nan_mask_y = torch.isnan(y_f32)
    if not torch.equal(nan_mask_x, nan_mask_y):
        raise AssertionError(
            f"[{label}] NaN mismatch: x has NaNs at {torch.where(nan_mask_x)}, y has NaNs at {torch.where(nan_mask_y)}")

    # 排除掉 NaN 的位置，避免后续计算报错
    valid_mask = ~nan_mask_x
    x_valid = x_f32[valid_mask]
    y_valid = y_f32[valid_mask]

    # 4. 检查 Inf
    inf_mask_x = torch.isinf(x_valid)
    inf_mask_y = torch.isinf(y_valid)
    if not torch.equal(inf_mask_x, inf_mask_y):
        raise AssertionError(f"[{label}] Inf mismatch.")

    # 如果两边都是 Inf (且符号相同)，视为相等，排除掉
    # 注意：这里简化处理，假设同号 Inf 相等。严格来说要判断 +Inf 和 -Inf
    valid_mask_no_inf = ~inf_mask_x
    x_final = x_valid[valid_mask_no_inf]
    y_final = y_valid[valid_mask_no_inf]

    if x_final.numel() == 0:
        return  # 所有元素都是 NaN 或 Inf 且位置匹配，通过

    # 5. 核心比对逻辑 (结合绝对误差和相对误差)
    # diff = |x - y|
    diff = torch.abs(x_final - y_final)
    # tolerance = atol + rtol * |y|
    tolerance = atol + (rtol * torch.abs(y_final))

    # 找出超出容差的元素
    failure_mask = diff > tolerance

    if torch.any(failure_mask):
        # 收集错误信息
        max_diff = diff.max().item()
        max_idx = torch.argmax(diff).item()

        # 计算实际的最大相对误差 (仅供参考)
        # 避免除以 0
        y_safe = y_final.clone()
        y_safe[y_safe == 0] = 1e-6
        max_rel_err = (diff / torch.abs(y_safe)).max().item()

        error_msg = (
            f"\n[{label}] Mismatch found!\n"
            f"Type: {x.dtype}\n"
            f"Max Abs Diff: {max_diff:.6f} (Threshold: {atol})\n"
            f"Max Rel Diff: {max_rel_err:.6f} (Threshold: {rtol})\n"
            f"First failure at index {max_idx} (in flattened valid array):\n"
            f"  x: {x_final[max_idx].item()}\n"
            f"  y: {y_final[max_idx].item()}\n"
        )
        raise AssertionError(error_msg)


def simplified_mx_quantize(fp_array: np.ndarray, mx_ele_dtype: str = "float8_e4m3fn") -> tuple:
    """
    简化的 MX 量化函数。
    输入: fp_array (shape=[n, h], dtype=fp16/bf16)
    输出: (scale_array, ele_array)
    """
    try:
        from ml_dtypes import float8_e5m2, float8_e4m3fn
        from en_dtypes import float8_e8m0
    except ImportError:
        raise AssertionError("Unsupported UT testcase due to lack of package ml_dtypes or en_dtypes")

        # --- 1. 参数与常量定义 ---
    BLOCK_SIZE = 32
    AXIS = -1  # 总是处理最后一个维度

    if mx_ele_dtype == "float8_e5m2":
        max_norm = 57344.0
        exp_bits, mantissa_bits = 5, 2
        target_dtype = float8_e5m2
    elif mx_ele_dtype == "float8_e4m3fn":
        max_norm = 448.0
        exp_bits, mantissa_bits = 4, 3
        target_dtype = float8_e4m3fn
    else:
        raise ValueError(f"Unsupported mx_ele_dtype: {mx_ele_dtype}")

    # --- 2. Padding & Reshape (分块) ---
    # 将 [N, H] -> [N, H_blocks, 32]
    orig_shape = fp_array.shape
    h_dim = orig_shape[AXIS]

    # 计算需要补齐的长度
    pad_len = (BLOCK_SIZE - (h_dim % BLOCK_SIZE)) % BLOCK_SIZE
    if pad_len > 0:
        # 仅在最后一个维度补 0
        pad_width = [(0, 0)] * fp_array.ndim
        pad_width[AXIS] = (0, pad_len)
        fp_array = np.pad(fp_array, pad_width, 'constant')

    padded_shape = fp_array.shape
    # Reshape 为 (..., blocks, block_size)
    new_shape = list(padded_shape)
    new_shape[AXIS] = new_shape[AXIS] // BLOCK_SIZE
    new_shape.append(BLOCK_SIZE)
    fp_array_blocked = fp_array.reshape(new_shape)

    # --- 3. 计算共享 Scale (Shared Exponent) ---
    # 逻辑: scale = floor(log2(max(abs(block)))) - ele_emax
    ele_emax = int(np.log2(max_norm))
    # 在 block 内部 (最后一个维度) 找最大值
    fp_abs_max = np.max(np.abs(fp_array_blocked), axis=-1, keepdims=True)

    # 避免 log2(0)
    FP32_MIN_NORMAL = 2 ** (-126)
    share_exp = np.floor(
        np.log2(fp_abs_max + FP32_MIN_NORMAL * (fp_abs_max == 0))) - ele_emax

    # 处理特殊值与截断 (E8M0 范围)
    share_exp[fp_abs_max == 0] = -float("inf")
    SCALE_EMAX = 127
    share_exp[share_exp > SCALE_EMAX] = float("NaN")
    share_exp[share_exp < -SCALE_EMAX] = -SCALE_EMAX

    # --- 4. 量化元素 (Quantize Elements) ---
    # 公式: scaled = input / 2^scale
    scale_val = 2.0 ** share_exp
    scaled_input = fp_array_blocked / scale_val

    # 模拟 FP8 的精度损失 (Round & Clamp)
    # 计算私有指数 private_exp
    min_exp = -(2 ** (exp_bits - 1)) + 2
    abs_scaled = np.abs(scaled_input)
    # 避免 log2(0)
    private_exp = np.floor(
        np.log2(abs_scaled + (abs_scaled == 0))).astype(np.int32)
    # private_exp = np.clip(private_exp, min=min_exp, max=None)
    private_exp = np.clip(private_exp, min_exp, None)

    # 对尾数进行舍入 (Round to Nearest Even / Rint)
    step_scale = 2.0 ** (mantissa_bits - private_exp)
    ret = scaled_input * step_scale
    ret = np.rint(ret)
    ret = ret / step_scale

    # 截断到最大范数
    ret = np.clip(ret, -max_norm, max_norm)

    # --- 5. 还原形状与格式转换 ---
    # 还原为 [N, H_padded]
    ele_array = ret.reshape(padded_shape)
    # 去除 Padding
    if pad_len > 0:
        ele_array = ele_array[..., :h_dim]

    # 处理 Scale 数组形状
    # share_exp 当前形状 (N, H_blocks, 1)，去掉最后的 1
    share_exp = np.squeeze(share_exp, axis=-1)
    scale_array = 2.0 ** share_exp

    # Scale 数组必须对齐到偶数 (Cube 硬件要求)
    if scale_array.shape[-1] % 2 != 0:
        scale_array = np.pad(scale_array, ((0, 0), (0, 1)),
                             'constant', constant_values=2**-127)

    # Reshape Scale 为 (N, H_blocks/2, 2)
    s_shape = list(scale_array.shape)
    s_shape[-1] //= 2
    s_shape.append(2)
    scale_array = scale_array.reshape(s_shape)

    # --- 6. 最终类型转换 ---
    # 转换 Scale 为 E8M0
    if float8_e8m0:
        scale_array = scale_array.astype(float8_e8m0)

    # 转换 Element 为目标 FP8
    # 先转 float32 确保兼容性 (特别是输入为 bf16 时)
    ele_array = ele_array.astype(np.float32)
    ele_array = np.nan_to_num(ele_array, nan=0.0)
    if target_dtype:
        ele_array = ele_array.astype(target_dtype)

    return scale_array, ele_array


class MoeInitRoutingV2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, expert_idx, *, scale=None, offset=None,
                active_num=-1, expert_capacity=-1, expert_num=-1, drop_pad_mode=0,
                expert_tokens_num_type=0, expert_tokens_num_flag=False,
                quant_mode=0, active_expert_range=0, row_idx_type=0):
        ddd = dict(x=x, expert_idx=expert_idx, scale=scale, offset=offset,
                   active_num=active_num, expert_capacity=expert_capacity, expert_num=expert_num,
                   drop_pad_mode=drop_pad_mode,
                   expert_tokens_num_type=expert_tokens_num_type, expert_tokens_num_flag=expert_tokens_num_flag,
                   quant_mode=quant_mode, active_expert_range=active_expert_range, row_idx_type=row_idx_type)
        for k, v in ddd.items():
            print(k, v)
        expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale = \
            torch.ops.npu.npu_moe_init_routing_v2(
                x, expert_idx, scale=scale, offset=offset,
                active_num=active_num, expert_capacity=expert_capacity, expert_num=expert_num,
                drop_pad_mode=drop_pad_mode,
                expert_tokens_num_type=expert_tokens_num_type, expert_tokens_num_flag=expert_tokens_num_flag,
                active_expert_range=active_expert_range, quant_mode=quant_mode, row_idx_type=row_idx_type
            )
        return expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale


class TestNpuMoeInitRoutingV2(TestCase):

    def adapter_capacity(self, sorted_row_idx, sorted_expert_idx, capacity):
        count = 0
        last = sorted_expert_idx[0]
        for i, val in enumerate(sorted_expert_idx):
            if last != val:
                count = 1
                last = val
            else:
                count += 1
                if count > capacity:
                    sorted_expert_idx[i] = -1
                    sorted_row_idx[i] = -1

    def cpu_op_exec(self, x, expert_idx, scale, offset, expert_range, quant_mode, row_idx_type, expert_tokens_num_flag, expert_tokens_num_type, drop_pad_mode, active_num, expert_capacity):
        expert_start = expert_range[0]
        expert_end = expert_range[1]
        expert_num = 32
        num_rows = x.shape[0]
        h = x.shape[1]
        k = expert_idx.shape[-1]
        expert_idx_in = expert_idx.copy().reshape(-1)
        actual_expert_total_num = np.sum(
            (expert_idx >= expert_start) & (expert_idx < expert_end))

        expert_idx_in[(expert_idx_in < expert_start)
                      ] = np.int32(np.iinfo(np.int32).max)
        sorted_expert_indices = np.argsort(
            expert_idx_in, axis=-1, kind="stable")
        sorted_expert_idx = expert_idx_in[sorted_expert_indices]
        if row_idx_type == 1:
            expanded_row_idx = sorted_expert_indices.astype(np.int32)
        else:
            expanded_row_idx = np.ones(num_rows * k).astype(np.int32) * -1
            tmp_indices = np.arange(actual_expert_total_num)
            expanded_row_idx[sorted_expert_indices[:actual_expert_total_num]] = tmp_indices

        if not expert_tokens_num_flag:
            expert_tokens_count = None
        else:
            if drop_pad_mode == 0:
                if expert_tokens_num_type == 1:
                    expert_tokens_count = np.bincount(
                        sorted_expert_idx[:actual_expert_total_num] - expert_start)
                    expert_tokens_count = np.concatenate(
                        [expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
                elif expert_tokens_num_type == 0:
                    expert_tokens_count = np.bincount(
                        sorted_expert_idx[:actual_expert_total_num] - expert_start)
                    expert_tokens_count = np.concatenate(
                        [expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
                    expert_tokens_count = np.cumsum(expert_tokens_count)
                elif expert_tokens_num_type == 2:
                    expert_id, counts = np.unique(
                        sorted_expert_idx[:actual_expert_total_num], return_counts=True)
                    expert_tokens_count = np.column_stack((expert_id, counts))
                    if expert_tokens_count.shape[0] < expert_num:
                        expert_tokens_count = np.concatenate(
                            (expert_tokens_count, [[0, 0],]), axis=0)
            else:
                expert_tokens_count = np.bincount(
                    sorted_expert_idx[:actual_expert_total_num] - expert_start)
                expert_tokens_count = np.concatenate(
                    [expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
            expert_tokens_count = expert_tokens_count.astype(np.int64)

        if drop_pad_mode == 0:
            if active_num == 0:
                active_num = actual_expert_total_num
            else:
                active_num = min(active_num, actual_expert_total_num)
            expanded_scale = None
            expanded_x = x[sorted_expert_indices[:active_num] // k, :]
            if scale is not None and quant_mode == -1:
                expanded_scale = scale[sorted_expert_indices[:active_num] // k]
        else:
            self.adapter_capacity(sorted_expert_indices,
                                  sorted_expert_idx, expert_capacity)

            sort_row_tmp = np.full(
                (expert_num * expert_capacity), -1, dtype=int)
            offset_tmp = 0
            lastExpertId = 0
            for i, val in enumerate(sorted_expert_indices):
                if val != -1:
                    if lastExpertId != sorted_expert_idx[i]:
                        offset_tmp = 0
                        lastExpertId = sorted_expert_idx[i]
                    sort_row_tmp[sorted_expert_idx[i] * expert_capacity +
                                 offset_tmp] = sorted_expert_indices[i]
                    offset_tmp = offset_tmp + 1

            expanded_row_idx = np.full(sorted_expert_indices.shape, -1)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_row_idx[val] = i

            expanded_x_mask = np.full(
                (expert_num * expert_capacity, h), 1, dtype=int)
            expanded_x = np.full(
                (expert_num * expert_capacity, h), 0, dtype=x.dtype)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_x[i] = x[val // k]
                    expanded_x_mask[i] = np.full((h,), 0, dtype=int)

        if quant_mode == -1:
            expanded_x = expanded_x
            expanded_row_idx = expanded_row_idx
            if scale is not None and drop_pad_mode == 1:
                expanded_scale = np.full(
                    (expert_num * expert_capacity,), 0, dtype=scale.dtype)
                for i, val in enumerate(sort_row_tmp):
                    if val != -1:
                        expanded_scale[i] = scale[val // k]
            if scale is None:
                expanded_scale = None

        if quant_mode == 0:
            expanded_scale = None
            expanded_x_fp16 = expanded_x.astype(np.float16)
            scale_val = scale.astype(np.float16)
            offset_val = offset.astype(np.float16)
            scale_rst = expanded_x_fp16 * scale_val[0]
            add_offset = scale_rst + offset_val[0]
            round_data = np.rint(add_offset)
            round_data = np.clip(round_data, -128, 127)
            expanded_x = round_data.astype(np.int8)

        if quant_mode == 1:
            x_final = expanded_x.astype(np.float32)
            if scale is None:
                x_abs = np.abs(x_final)
                x_max = np.max(x_abs, axis=-1, keepdims=True)
                expanded_scale = x_max / 127
                expanded_x = x_final / expanded_scale
                expanded_x = np.round(expanded_x).astype(np.int8)
            else:
                if scale.shape[0] == 1:
                    x_final = x_final * scale
                else:
                    if drop_pad_mode == 0:
                        x_final = x_final * \
                            scale[sorted_expert_idx[:active_num] - expert_start]
                    else:
                        for i, val in enumerate(sort_row_tmp):
                            if val != -1:
                                x_final[i] = x_final[i] * \
                                    scale[i // expert_capacity]
                x_abs = np.abs(x_final)
                x_max = np.max(x_abs, axis=-1, keepdims=True)
                expanded_scale = x_max / 127
                expanded_x = x_final / expanded_scale
                expanded_x = np.round(expanded_x).astype(np.int8)
            if x.dtype == np.int8:
                expanded_scale == None

        if quant_mode == 2 or quant_mode == 3:
            quant_mode_dtype_str_map = {2: "float8_e5m2", 3: "float8_e4m3fn"}
            expanded_scale, expanded_x = simplified_mx_quantize(
                expanded_x, mx_ele_dtype=quant_mode_dtype_str_map[quant_mode])
            ess = expanded_scale.shape
            expanded_scale = expanded_scale.reshape(
                *ess[:-2], ess[-2] * ess[-1])

        if drop_pad_mode == 1:
            expanded_x = np.ma.array(
                expanded_x, mask=expanded_x_mask).filled(0)
            expanded_x = expanded_x.reshape(expert_num, expert_capacity, h)

        return expanded_x, expanded_row_idx.astype(np.int32), expert_tokens_count, expanded_scale

    def npu_op_exec(self, x, expert_idx, scale, offset, active_expert_range, quant_mode, row_idx_type, expert_tokens_num_flag, expert_tokens_num_type, drop_pad_mode, active_num, expert_capacity):
        bs = x.shape[0]
        k = expert_idx.shape[1]
        expert_num = 32
        expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale = torch_npu.npu_moe_init_routing_v2(
            x, expert_idx, scale=scale, offset=offset,
            active_num=active_num, expert_capacity=expert_capacity, expert_num=expert_num,
            drop_pad_mode=drop_pad_mode,
            expert_tokens_num_type=expert_tokens_num_type, expert_tokens_num_flag=expert_tokens_num_flag,
            active_expert_range=active_expert_range, quant_mode=quant_mode, row_idx_type=row_idx_type
        )
        return expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale

    def assertExpandedXRtolEqual(self, expanded_x, local_expanded_x_npu, dtype):
        if dtype == torch.bfloat16:
            self.assertRtolEqual(torch.tensor(
                expanded_x, dtype=torch.bfloat16), local_expanded_x_npu)
        elif dtype == np.int8:
            self.assertEqual(expanded_x.shape, local_expanded_x_npu.shape)
            self.assertEqual(np.int8, local_expanded_x_npu.numpy().dtype)
            max_diff = np.abs(expanded_x - local_expanded_x_npu.numpy()).max()
            self.assertLessEqual(max_diff, 1)
        else:
            self.assertRtolEqual(expanded_x, local_expanded_x_npu.numpy())

    def generate_inputs(self, bs, h, k, dtype, scale_shape, none_scale, none_offset, drop_pad_mode):
        if dtype == torch.bfloat16:
            x = np.random.uniform(-1, 1, size=(bs, h)).astype(np.float32)
            x_npu = torch.tensor(x, dtype=torch.bfloat16).npu()
        elif dtype == np.int8:
            x = np.random.uniform(-127, 128, size=(bs, h)).astype(dtype)
            x_npu = torch.from_numpy(x).npu()
        else:
            x = np.random.uniform(-1, 1, size=(bs, h)).astype(dtype)
            x_npu = torch.from_numpy(x).npu()
        expert_idx = np.random.randint(0, 32, size=(bs, k)).astype(np.int32)
        scale = None if none_scale else np.random.uniform(
            -1, 1, size=scale_shape).astype(np.float32)
        offset = None if none_offset or none_scale else np.random.uniform(
            -1, 1, size=scale_shape).astype(np.float32)

        expert_idx_npu = torch.from_numpy(expert_idx).npu()
        scale_npu = None if scale is None else torch.from_numpy(
            scale).contiguous().npu()
        offset_npu = None if offset is None else torch.from_numpy(
            offset).contiguous().npu()

        expert_tokens_num_type = 1 if drop_pad_mode == 1 else random.choice([
                                                                            0, 1, 2])
        row_idx_type = 0 if drop_pad_mode == 1 else random.choice([0, 1])
        # active_num = 0 if drop_pad_mode == 1 else random.randint(0, bs * k)
        active_num = bs * k
        expert_capacity = -1 if drop_pad_mode == 0 else random.randint(1, bs)

        return x, expert_idx, scale, offset, x_npu, expert_idx_npu, scale_npu, offset_npu, expert_tokens_num_type, row_idx_type, active_num, expert_capacity

    def calc_npu_vs_golden(self, x, expert_idx, scale, offset,
                           x_npu, expert_idx_npu, scale_npu, offset_npu,
                           expert_range, quant_mode, row_idx_type, expert_tokens_num_flag, expert_tokens_num_type, drop_pad_mode, active_num, expert_capacity):
        expanded_x_npu, expanded_row_idx_npu, expert_tokens_count_npu, expanded_scale_npu = self.npu_op_exec(
            x_npu, expert_idx_npu, scale=scale_npu, offset=offset_npu,
            active_expert_range=expert_range, quant_mode=quant_mode, row_idx_type=row_idx_type, expert_tokens_num_flag=expert_tokens_num_flag, expert_tokens_num_type=expert_tokens_num_type,
            drop_pad_mode=drop_pad_mode, active_num=active_num, expert_capacity=expert_capacity)
        expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale = self.cpu_op_exec(
            x, expert_idx, scale=scale, offset=offset,
            expert_range=expert_range, quant_mode=quant_mode, row_idx_type=row_idx_type, expert_tokens_num_flag=expert_tokens_num_flag, expert_tokens_num_type=expert_tokens_num_type,
            drop_pad_mode=drop_pad_mode, active_num=active_num, expert_capacity=expert_capacity)

        local_expanded_x_npu = expanded_x_npu.cpu()
        local_expanded_row_idx_npu = expanded_row_idx_npu.cpu()
        local_expert_tokens_count_npu = expert_tokens_count_npu.cpu()
        local_expanded_scale_npu = expanded_scale_npu.cpu()

        actual_expert_total_num = np.sum(
            (expert_idx >= expert_range[0]) & (expert_idx < expert_range[1]))
        if drop_pad_mode == 0:
            if active_num == 0:
                actual_expert_count = actual_expert_total_num
            else:
                actual_expert_count = min(active_num, actual_expert_total_num)
            local_expanded_x_npu = local_expanded_x_npu[:actual_expert_count]
            local_expanded_scale_npu = local_expanded_scale_npu[:actual_expert_count]

        if expert_tokens_num_flag == True and expert_tokens_num_type == 2:
            length = expert_tokens_count.shape[0]
            local_expert_tokens_count_npu = local_expert_tokens_count_npu[:length]

        return expanded_x, local_expanded_x_npu, expanded_row_idx, local_expanded_row_idx_npu, \
            expert_tokens_count, local_expert_tokens_count_npu, expanded_scale, local_expanded_scale_npu

    @SupportedDevices(['Ascend910B', 'Ascend910_95', 'Ascend950'])
    def test_npu_moe_init_routing_no_quant(self):
        bs_list = [32]
        h_list = [14, 200]
        k_list = [5]
        expert_range_list = [[0, 16]]
        quant_mode_list = [-1]
        drop_pad_mode_list = [0, 1]
        row_idx_type_list = [0, 1]
        expert_tokens_num_flags = [True, False]
        dtype_list = [np.int8, np.float16, np.float32, torch.bfloat16]
        none_scales = [True, False]
        none_offsets = [True]
        for bs, h, k, expert_range, quant_mode, row_idx_type, dtype, none_scale, none_offset, expert_tokens_num_flag, drop_pad_mode in itertools.product(
                bs_list, h_list, k_list, expert_range_list, quant_mode_list, row_idx_type_list,
                dtype_list, none_scales, none_offsets, expert_tokens_num_flags, drop_pad_mode_list):
            scale_shape = (bs,)
            expert_range = [0, 32] if drop_pad_mode == 1 else [0, 16]
            x, expert_idx, scale, offset, x_npu, expert_idx_npu, scale_npu, offset_npu, expert_tokens_num_type, row_idx_type, active_num, expert_capacity = self.generate_inputs(
                bs, h, k, dtype, scale_shape, none_scale, none_offset, drop_pad_mode)
            if drop_pad_mode == 1 or expert_tokens_num_flag == False or expert_tokens_num_type == 0:
                continue
            expanded_x, local_expanded_x_npu, expanded_row_idx, local_expanded_row_idx_npu, \
                expert_tokens_count, local_expert_tokens_count_npu, expanded_scale, local_expanded_scale_npu \
                = self.calc_npu_vs_golden(x, expert_idx, scale, offset,
                                          x_npu, expert_idx_npu, scale_npu, offset_npu,
                                          expert_range, quant_mode, row_idx_type, expert_tokens_num_flag, expert_tokens_num_type, drop_pad_mode, active_num, expert_capacity)

            self.assertExpandedXRtolEqual(
                expanded_x, local_expanded_x_npu, dtype)
            self.assertRtolEqual(
                expanded_row_idx, local_expanded_row_idx_npu.numpy())
            if expert_tokens_num_flag:
                self.assertRtolEqual(expert_tokens_count,
                                     local_expert_tokens_count_npu.numpy())
            if none_scale:
                return
            self.assertRtolEqual(
                expanded_scale, local_expanded_scale_npu.numpy())

    @unittest.skip("Skipping test_npu_moe_init_routing_static_quant for now")
    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_init_routing_static_quant(self):
        bs_list = [32]
        h_list = [14, 200]
        k_list = [5, 128]
        expert_range_list = [[0, 16]]
        quant_mode_list = [0]
        drop_pad_mode_list = [0, 1]
        row_idx_type_list = [0, 1]
        expert_tokens_num_flags = [True, False]
        dtype_list = [np.float16, np.float32, torch.bfloat16]
        none_scales = [False]
        none_offsets = [False]
        for bs, h, k, expert_range, quant_mode, row_idx_type, dtype, none_scale, none_offset, expert_tokens_num_flag, drop_pad_mode in itertools.product(
                bs_list, h_list, k_list, expert_range_list, quant_mode_list, row_idx_type_list,
                dtype_list, none_scales, none_offsets, expert_tokens_num_flags, drop_pad_mode_list):
            scale_shape = (1,)
            expert_range = [0, 32] if drop_pad_mode == 1 else [0, 16]
            x, expert_idx, scale, offset, x_npu, expert_idx_npu, scale_npu, offset_npu, expert_tokens_num_type, row_idx_type, active_num, expert_capacity = self.generate_inputs(
                bs, h, k, dtype, scale_shape, none_scale, none_offset, drop_pad_mode)

            expanded_x, local_expanded_x_npu, expanded_row_idx, local_expanded_row_idx_npu, \
                expert_tokens_count, local_expert_tokens_count_npu, expanded_scale, local_expanded_scale_npu \
                = self.calc_npu_vs_golden(x, expert_idx, scale, offset,
                                          x_npu, expert_idx_npu, scale_npu, offset_npu,
                                          expert_range, quant_mode, row_idx_type, expert_tokens_num_flag, expert_tokens_num_type, drop_pad_mode, active_num, expert_capacity)

            self.assertExpandedXRtolEqual(
                expanded_x, local_expanded_x_npu, np.int8)
            self.assertRtolEqual(
                expanded_row_idx, local_expanded_row_idx_npu.numpy())
            if expert_tokens_num_flag:
                self.assertRtolEqual(expert_tokens_count,
                                     local_expert_tokens_count_npu.numpy())

    @SupportedDevices(['Ascend910B', 'Ascend910_95', 'Ascend950'])
    def test_npu_moe_init_routing_dynamic_quant(self):
        bs_list = [32]
        h_list = [14, 200]
        k_list = [8]
        expert_range_list = [[0, 16]]
        quant_mode_list = [1]
        drop_pad_mode_list = [0, 1]
        row_idx_type_list = [0, 1]
        expert_tokens_num_flags = [True, False]
        dtype_list = [np.float16, np.float32, torch.bfloat16]
        none_scales = [True, False]
        none_offsets = [True]
        for bs, h, k, expert_range, quant_mode, row_idx_type, dtype, none_scale, none_offset, expert_tokens_num_flag, drop_pad_mode in itertools.product(
                bs_list, h_list, k_list, expert_range_list, quant_mode_list, row_idx_type_list,
                dtype_list, none_scales, none_offsets, expert_tokens_num_flags, drop_pad_mode_list):
            expert_range_length = expert_range[1] - expert_range[0]
            scale_shape = (expert_range_length, h)
            expert_range = [0, 32] if drop_pad_mode == 1 else [0, 16]
            x, expert_idx, scale, offset, x_npu, expert_idx_npu, scale_npu, offset_npu, expert_tokens_num_type, row_idx_type, active_num, expert_capacity = self.generate_inputs(
                bs, h, k, dtype, scale_shape, none_scale, none_offset, drop_pad_mode)
            if drop_pad_mode == 1 or expert_tokens_num_flag == False or expert_tokens_num_type == 0:
                continue
            expanded_x, local_expanded_x_npu, expanded_row_idx, local_expanded_row_idx_npu, \
                expert_tokens_count, local_expert_tokens_count_npu, expanded_scale, local_expanded_scale_npu \
                = self.calc_npu_vs_golden(x, expert_idx, scale, offset,
                                          x_npu, expert_idx_npu, scale_npu, offset_npu,
                                          expert_range, quant_mode, row_idx_type, expert_tokens_num_flag, expert_tokens_num_type, drop_pad_mode, active_num, expert_capacity)

            self.assertExpandedXRtolEqual(
                expanded_x, local_expanded_x_npu, np.int8)
            self.assertRtolEqual(
                expanded_row_idx, local_expanded_row_idx_npu.numpy())
            if expert_tokens_num_flag:
                self.assertRtolEqual(expert_tokens_count,
                                     local_expert_tokens_count_npu.numpy())
            if none_scale:
                return
            self.assertRtolEqual(
                expanded_scale, local_expanded_scale_npu.numpy())

    @unittest.skipIf(
        torch.__version__ < "2.7" or
        importlib.util.find_spec("ml_dtypes") is None or
        importlib.util.find_spec("en_dtypes") is None,
        "Unittest for mxfp8 need torch.version>=2.7 and package ml_dtypes, en_dtypes"
    )
    @SupportedDevices(['Ascend910_95', 'Ascend950'])
    def test_npu_moe_init_routing_mxfp8_quant(self):
        bs_list = [32]
        h_list = [7168, 7184]
        k_list = [8]
        expert_range_list = [[0, 32]]
        quant_mode_list = [2, 3]
        drop_pad_mode_list = [0]
        row_idx_type_list = [0, 1]
        expert_tokens_num_type = 1
        expert_tokens_num_flags = [True]
        dtype_list = [torch.float16, torch.bfloat16]
        none_scales = [True, False]
        none_offsets = [True]
        for bs, h, k, expert_range, quant_mode, row_idx_type, x_dtype, none_scale, none_offset, expert_tokens_num_flag, drop_pad_mode in itertools.product(
                bs_list, h_list, k_list, expert_range_list, quant_mode_list, row_idx_type_list,
                dtype_list, none_scales, none_offsets, expert_tokens_num_flags, drop_pad_mode_list):
            # generate inputs
            scale_shape = (bs*k, h)
            x, expert_idx, scale, offset, x_npu, expert_idx_npu, scale_npu, offset_npu, _, _, active_num, expert_capacity = self.generate_inputs(
                bs, h, k, x_dtype, scale_shape, none_scale, none_offset, drop_pad_mode)
            # run npu&cpu
            expanded_x, local_expanded_x_npu, expanded_row_idx, local_expanded_row_idx_npu, \
                expert_tokens_count, local_expert_tokens_count_npu, expanded_scale, local_expanded_scale_npu \
                = self.calc_npu_vs_golden(x, expert_idx, scale, offset,
                                          x_npu, expert_idx_npu, scale_npu, offset_npu,
                                          expert_range, quant_mode, row_idx_type, expert_tokens_num_flag, expert_tokens_num_type, drop_pad_mode, active_num, expert_capacity)
            # compare
            assert_tensors_close(numpy_to_torch(
                expanded_x), local_expanded_x_npu)
            assert_tensors_close(numpy_to_torch(
                expanded_row_idx), local_expanded_row_idx_npu)
            assert_tensors_close(numpy_to_torch(
                expert_tokens_count), local_expert_tokens_count_npu)
            assert_tensors_close(numpy_to_torch(
                expanded_scale), local_expanded_scale_npu)


if __name__ == "__main__":
    run_tests()
