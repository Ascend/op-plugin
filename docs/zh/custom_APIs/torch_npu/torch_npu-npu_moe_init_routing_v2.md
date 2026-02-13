# torch_npu.npu_moe_init_routing_v2<a name="ZH-CN_TOPIC_0000002309015148"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>  | √   |

## 功能说明<a name="zh-cn_topic_0000002271534921_section1650913464367"></a>

-   API功能：MoE（Mixture of Experts）的routing计算，根据[torch_npu.npu_moe_gating_top_k_softmax](torch_npu-npu_moe_gating_top_k_softmax.md)的计算结果做routing处理，支持不量化、动态量化和静态量化模式。
- 计算公式：  

  1.对输入expertIdx做排序，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx：

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

  2.以sortedRowIdx做位置映射得出expandedRowIdxOut：
    - rowIdxType等于1时, 输出scatter索引
      $$
      expandedRowIdxOut[i]=sortedRowIdx[i]
      $$
    - rowIdxType等于0时, 输出gather索引
      $$
      expandedRowIdxOut[sortedRowIdx[i]]=i
      $$
      
  3.对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensCountOrCumsumOutOptional：

    $$
    expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
    $$

  4.如果quantMode不等于-1, 计算quant结果：
     - 静态quant
     $$
     quantResult=round((x∗scaleOptional)+offsetOptional)
     $$
     
    - 动态quant：
        - 若不输入scale：
            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$
        - 若输入scale:
            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$
  
  5.若活跃的expert范围为全专家范围时，按照Scatter索引搬运token；反之按照Gather索引搬运token。在dropPadMode为1时将每个专家需要处理的Token个数对齐为expertCapacity个，超过expertCapacity个的Token会被Drop，不足的会用0填充。得出expandedXOut：
    - 非量化场景
      - 按照Scatter索引搬运
      $$
      expandedXOut[i]=x[scatterRowIdx[i]//K]
      $$
      - 按照Gather索引搬运
      $$
      expandedXOut[gatherRowIdx[i]]=x[i//K]
      $$
    - 量化场景
      - 按照Scatter索引搬运
      $$
      expandedXOut[i]=quantResult[scatterRowIdx[i]//K]
      $$
      - 按照Gather索引搬运
      $$
      expandedXOut[gatherRowIdx[i]]=quantResult[i//K]
      $$

  6.expandedRowIdxOut的有效元素数量availableIdxNum，计算方式为expertIdx中activeExpertRangeOptional范围内的元素的个数
    $$
    availableIdxNum = |\{x\in expertIdx| expert\_start \le x<expert\_end \ \}|
    $$

-   等价计算逻辑
    ```python
    import numpy as np
    import random


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


    class MoeInitRoutingV2CPU:
        """
        CPU上MoeInitRoutingV2的等价实现
        """

        @staticmethod
        def adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity):
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

        @staticmethod
        def cpu_op_exec(x, expert_idx, scale=None, offset=None, expert_range=None, 
                        quant_mode=-1, row_idx_type=0, expert_tokens_num_flag=False, 
                        expert_tokens_num_type=0, drop_pad_mode=0, active_num=-1, 
                        expert_capacity=-1):
            """
            CPU上MoeInitRoutingV2的等价计算逻辑
            
            参数:
                x: 输入tensor, shape (bs, h)
                expert_idx: 专家索引, shape (bs, k)
                scale: 量化scale, 可选
                offset: 量化offset, 可选
                expert_range: 专家范围 [start, end], 默认 [0, 16]
                quant_mode: 量化模式 (-1:无量化, 0:静态量化, 1:动态量化, 2/3:MXFP8)
                row_idx_type: row_idx类型 (0或1)
                expert_tokens_num_flag: 是否计算专家token数量
                expert_tokens_num_type: 专家token数量类型 (0,1,2)
                drop_pad_mode: drop_pad模式 (0或1)
                active_num: 激活数量
                expert_capacity: 专家容量
                
            返回:
                expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale
            """
            if expert_range is None:
                expert_range = [0, 16]
                
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
                if active_num == 0 or active_num == -1:
                    active_num = actual_expert_total_num
                else:
                    active_num = min(active_num, actual_expert_total_num)
                expanded_scale = None
                expanded_x = x[sorted_expert_indices[:active_num] // k, :]
                if scale is not None and quant_mode == -1:
                    expanded_scale = scale[sorted_expert_indices[:active_num] // k]
            else:
                MoeInitRoutingV2CPU.adapter_capacity(sorted_expert_indices,
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
                    expanded_scale = None

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

        @staticmethod
        def generate_inputs(bs, h, k, dtype, scale_shape, none_scale, none_offset, drop_pad_mode):
            """
            生成测试输入数据
            """
            if dtype == np.float16 or dtype == 'float16':
                x = np.random.uniform(-1, 1, size=(bs, h)).astype(np.float16)
            elif dtype == np.int8 or dtype == 'int8':
                x = np.random.uniform(-127, 128, size=(bs, h)).astype(np.int8)
            else:
                x = np.random.uniform(-1, 1, size=(bs, h)).astype(np.float32)
                
            expert_idx = np.random.randint(0, 32, size=(bs, k)).astype(np.int32)
            scale = None if none_scale else np.random.uniform(
                -1, 1, size=scale_shape).astype(np.float32)
            offset = None if none_offset or none_scale else np.random.uniform(
                -1, 1, size=scale_shape).astype(np.float32)

            expert_tokens_num_type = 1 if drop_pad_mode == 1 else random.choice([0, 1, 2])
            row_idx_type = 0 if drop_pad_mode == 1 else random.choice([0, 1])
            active_num = bs * k
            expert_capacity = -1 if drop_pad_mode == 0 else random.randint(1, bs)

            return x, expert_idx, scale, offset, expert_tokens_num_type, row_idx_type, active_num, expert_capacity

    def demo_no_quant():
        """无量化模式"""
        print("=" * 50)
        print("Demo: 无量化模式 (quant_mode=-1)")
        print("=" * 50)
        
        bs, h, k = 32, 200, 5
        expert_range = [0, 16]
        
        # 生成输入
        x, expert_idx, scale, offset, expert_tokens_num_type, row_idx_type, active_num, expert_capacity = \
            MoeInitRoutingV2CPU.generate_inputs(bs, h, k, np.float16, (bs,), False, True, 0)
        
        # 执行计算
        expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale = \
            MoeInitRoutingV2CPU.cpu_op_exec(
                x, expert_idx, scale, offset,
                expert_range=expert_range,
                quant_mode=-1,
                row_idx_type=row_idx_type,
                expert_tokens_num_flag=True,
                expert_tokens_num_type=expert_tokens_num_type,
                drop_pad_mode=0,
                active_num=active_num,
                expert_capacity=expert_capacity
            )
        
        print(f"Input x shape: {x.shape}, dtype: {x.dtype}")
        print(f"Input expert_idx shape: {expert_idx.shape}")
        print(f"Output expanded_x shape: {expanded_x.shape}, dtype: {expanded_x.dtype}")
        print(f"Output expanded_row_idx shape: {expanded_row_idx.shape}")
        print(f"Output expert_tokens_count: {expert_tokens_count}")
        print(f"Output expanded_scale: {expanded_scale}")
        print()


    def demo_static_quant():
        """静态量化模式"""
        print("=" * 50)
        print("Demo: 静态量化模式 (quant_mode=0)")
        print("=" * 50)
        
        bs, h, k = 32, 200, 5
        expert_range = [0, 16]
        
        # 生成输入 (需要scale和offset)
        x, expert_idx, scale, offset, expert_tokens_num_type, row_idx_type, active_num, expert_capacity = \
            MoeInitRoutingV2CPU.generate_inputs(bs, h, k, np.float32, (1,), False, False, 0)
        
        # 执行计算
        expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale = \
            MoeInitRoutingV2CPU.cpu_op_exec(
                x, expert_idx, scale, offset,
                expert_range=expert_range,
                quant_mode=0,
                row_idx_type=row_idx_type,
                expert_tokens_num_flag=True,
                expert_tokens_num_type=1,
                drop_pad_mode=0,
                active_num=active_num,
                expert_capacity=expert_capacity
            )
        
        print(f"Input x shape: {x.shape}, dtype: {x.dtype}")
        print(f"Input scale shape: {scale.shape}, offset shape: {offset.shape}")
        print(f"Output expanded_x shape: {expanded_x.shape}, dtype: {expanded_x.dtype}")
        print(f"Output expanded_row_idx shape: {expanded_row_idx.shape}")
        print(f"Output expert_tokens_count: {expert_tokens_count}")
        print()


    def demo_dynamic_quant():
        """动态量化模式"""
        print("=" * 50)
        print("Demo: 动态量化模式 (quant_mode=1)")
        print("=" * 50)
        
        bs, h, k = 32, 200, 8
        expert_range = [0, 16]
        expert_range_length = expert_range[1] - expert_range[0]
        
        # 生成输入 (可选scale)
        x, expert_idx, scale, offset, expert_tokens_num_type, row_idx_type, active_num, expert_capacity = \
            MoeInitRoutingV2CPU.generate_inputs(bs, h, k, np.float32, (expert_range_length, h), False, True, 0)
        
        # 执行计算
        expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale = \
            MoeInitRoutingV2CPU.cpu_op_exec(
                x, expert_idx, scale, offset,
                expert_range=expert_range,
                quant_mode=1,
                row_idx_type=row_idx_type,
                expert_tokens_num_flag=True,
                expert_tokens_num_type=1,
                drop_pad_mode=0,
                active_num=active_num,
                expert_capacity=expert_capacity
            )
        
        print(f"Input x shape: {x.shape}, dtype: {x.dtype}")
        print(f"Input scale shape: {scale.shape}")
        print(f"Output expanded_x shape: {expanded_x.shape}, dtype: {expanded_x.dtype}")
        print(f"Output expanded_scale shape: {expanded_scale.shape if expanded_scale is not None else None}")
        print()


    def demo_drop_pad_mode():
        """drop_pad模式"""
        print("=" * 50)
        print("Demo: drop_pad_mode=1")
        print("=" * 50)
        
        bs, h, k = 32, 200, 5
        expert_range = [0, 32]  # drop_pad_mode=1时通常使用全范围
        
        # 生成输入
        x, expert_idx, scale, offset, expert_tokens_num_type, row_idx_type, active_num, expert_capacity = \
            MoeInitRoutingV2CPU.generate_inputs(bs, h, k, np.float16, (bs,), False, True, 1)
        
        # 执行计算
        expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale = \
            MoeInitRoutingV2CPU.cpu_op_exec(
                x, expert_idx, scale, offset,
                expert_range=expert_range,
                quant_mode=-1,
                row_idx_type=0,  # drop_pad_mode=1时row_idx_type强制为0
                expert_tokens_num_flag=True,
                expert_tokens_num_type=1,  # drop_pad_mode=1时强制为1
                drop_pad_mode=1,
                active_num=active_num,
                expert_capacity=expert_capacity
            )
        
        print(f"Input x shape: {x.shape}, dtype: {x.dtype}")
        print(f"Input expert_idx shape: {expert_idx.shape}")
        print(f"Output expanded_x shape: {expanded_x.shape}, dtype: {expanded_x.dtype}")
        print(f"Output expanded_row_idx shape: {expanded_row_idx.shape}")
        print(f"Output expert_tokens_count: {expert_tokens_count}")
        print()


    if __name__ == "__main__":
        demo_no_quant()
        demo_static_quant()
        demo_dynamic_quant()
        demo_drop_pad_mode()
    ```


## 函数原型<a name="zh-cn_topic_0000002271534921_section14509346133618"></a>

```
torch_npu.npu_moe_init_routing_v2(x, expert_idx, *, scale=None, offset=None, active_num=-1, expert_capacity=-1, expert_num=-1, drop_pad_mode=0, expert_tokens_num_type=0, expert_tokens_num_flag=False, quant_mode=-1, active_expert_range=[], row_idx_type=0) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002271534921_section2050919466367"></a>

-   **x** (`Tensor`)：必选参数，表示MoE的输入即token特征输入，要求为2维张量，shape为(NUM_ROWS, H)。数据类型支持`float16`、`bfloat16`、`float32`、`int8`，数据格式要求为$ND$。
-   **expert_idx** (`Tensor`)：必选参数，表示[torch_npu.npu_moe_gating_top_k_softmax](torch_npu-npu_moe_gating_top_k_softmax.md)输出每一行特征对应的K个处理专家，要求是2维张量，shape为(NUM_ROWS, K)，且专家id不能超过专家数。数据类型支持`int32`，数据格式要求为$ND$。
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
-   **scale** (`Tensor`)：可选参数，默认为None，用于计算量化结果的参数。数据类型支持`float32`，数据格式要求为$ND$。如果不输入表示计算时不使用`scale`，且输出`expanded_scale`中的值无意义。
    -   非量化场景下，如果输入则要求为1维张量，shape为(NUM_ROWS,)。
    -   静态量化场景必须输入，输入要求为1D的Tensor，shape为(1,)
    -   动态quant场景下，如果输入则要求为2维张量，shape为(expert_end-expert_start, H)或(1, H)。

-   **offset** (`Tensor`)：可选参数，默认为None，用于计算量化结果的偏移值。数据类型支持`float32`，数据格式要求为$ND$。
    -   在非量化场景下不输入。
    -   静态量化场景必须输入，输入要求为1维张量，shape为(1,)
    -   动态quant场景下不输入。

-   **active_num** (`int`)：可选参数，默认值为-1，表示总的最大处理row数，输出`expanded_x`只有这么多行是有效的，入参校验需大于等于0，0表示Dropless场景，大于0时表示Active场景，约束所有专家共同处理tokens总量。
-   **expert_capacity** (`int`)：可选参数，默认值为-1，表示每个专家能够处理的tokens数，入参校验大于0小于NUM_ROWS。
-   **expert_num** (`int`)：可选参数，默认值为-1，表示专家数。`expert_tokens_num_type`为key_value模式时，取值范围为[0, 5120]；其他模式取值范围为[0, 10240]。
-   **drop_pad_mode** (`int`)：可选参数，默认值为0，表示是否为drop_pad场景。0表示dropless场景，该场景下不校验`expert_capacity`。1表示drop_pad场景。
-   **expert_tokens_num_type** (`int`)：可选参数，默认值为0，表示直方图的不同模式。取值为0、1和2。0表示cumsum模式；1表示count模式，即输出的值为各个专家处理的token数量的累计值；2表示key_value模式，即输出的值为专家和对应专家处理token数量的累计值。
-   **expert_tokens_num_flag** (`bool`)：可选参数，默认值为False，取值为False和True，表示是否输出`expert_token_cumsum_or_count`。
-   **quant_mode** (`int`)：可选参数，默认值为-1，表示量化模式，支持取值为0、1、-1。0表示静态量化，-1表示不量化场景；1表示动态quant场景。
-   **active_expert_range** (`List[int]`)：可选参数，默认为空, 表示活跃expert的范围。数组内值的范围为[expert_start, expert_end]，左闭右开，表示活跃的expert范围在expert_start到expert_end之间。要求值大于等于0，并且expert_end不大于`expert_num`。drop_pad场景下，expert_start等于0, expert_end等于`expert_num`。传入默认值时，视为活跃的expert范围在0到`expert_num`之间。
-   **row_idx_type** (`int`)：可选参数，默认为0，表示输出`expanded_row_idx`使用的索引类型，支持取值0和1。0表示gather类型的索引；1表示scatter类型的索引。

## 返回值说明<a name="zh-cn_topic_0000002271534921_section18510124618368"></a>

-   **expanded_x** (`Tensor`)：根据`expert_idx`进行扩展过的特征，Dropless场景shape为[NUM_ROWS * K, H]。Active场景shape为[min(activeNum, NUM_ROWS * K), H]。Drop/Pad场景下要求是一个3D的Tensor，shape为[expertNum, expertCapacity, H]。非量化场景下数据类型同`x`；量化场景下数据类型为`int8`。数据格式要求为$ND$。量化场景下，当`x`的数据类型为`int8`时，输出值无意义。
-   **expanded_row_idx** (`Tensor`)：`expanded_x`和`x`的映射关系，要求是1维张量，shape为(NUM_ROWS\*K, )，数据类型支持`int32`，数据格式要求为$ND$。前available_idx_num\*H个元素为有效数据，其余由`row_idx_type`决定。当rowIdxType为0时，无效数据由-1填充；当rowIdxType为1时，无效数据未初始化。
-   **expert_token_cumsum_or_count** (`Tensor`)：表示输出每个专家处理的token数量的统计结果或累加值。
    -   在`expertTokensNumType`为0时，表示`active_expert_range`范围内expert在排序后处理token总数的前缀和。
    -   在`expert_tokens_num_type`为1的场景下，要求是1维张量，表示`active_expert_range`范围内expert对应的处理token的总数，shape为(expert_end-expert_start, )。shape为(expert_end-expert_start, )；
    -   在`expert_tokens_num_type`为2的场景下，要求是2维张量，shape为(expert_num, 2)，表示`active_expert_range`范围内token总数为非0的expert，以及对应expert处理token的总数；

    expert_idx在active_expert_range范围且剔除对应expert处理token为0的元素对为有效元素对，存放于Tensor头部并保持原序。数据类型支持`int64`，数据格式要求为$ND$。
-   **expanded_scale** (`Tensor`)：数据类型支持`float32`，数据格式要求为$ND$。输出shape为`expert_idx`的shape去掉最后一维之后所有维度的乘积。令available_idx_num为`active_expert_range`范围的元素的个数。
    -   非量化场景下，当`scale`输入时，前`available_idx_num`个元素为有效数据。
    -   动态quant场景下，输出量化计算过程中`scale`的中间值，前`available_idx_num`个元素为有效数据。
    -   静态量化场景下不输出。

## 约束说明<a name="zh-cn_topic_0000002271534921_section75102046193618"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   进入低时延性能模板需要同时满足以下条件：
    -   `x`、`expert_idx`、`scale`输入Shape要求分别为：(1, 7168)、(1, 8)、(256, 7168)
    -   `x`数据类型要求：`bfloat16`
    -   属性要求：active_expert_range=[0, 256]、 quant_mode=1、expert_tokens_num_type=2、expert_num=256

-   进入大batch性能模板需要同时满足以下条件：
    -   NUM_ROWS范围为[384, 8192]
    -   K=8
    -   expert_num=256
    -   expert_end-expert_start<=32
    -   quant_mode=-1
    -   row_idx_type=1
    -   expert_tokens_num_type=1

-   在算子输入shape较小的场景，操作间的多核同步时间占比较高，成为性能瓶颈。因此，针对这种特化场景，添加全载性能模板。该模板中，搬入、排序、计算都在同一个kernel内完成。需要满足 drop_pad_mode=0 的条件。

## 调用示例<a name="zh-cn_topic_0000002271534921_section12510194643618"></a>

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    
    bs = 1
    h = 613
    k = 475
    active_num = 475
    expert_capacity = -1
    expert_num = 226
    drop_pad_mode = 0
    expert_tokens_num_type = 1
    expert_tokens_num_flag = True
    quant_mode = -1
    active_expert_range = [23, 35]
    row_idx_type = 0
    
    x = torch.randn((bs, h), dtype=torch.float32).npu()
    expert_idx = torch.randint(0, expert_num, (bs, k), dtype=torch.int32).npu()
    scale = torch.randn((bs,), dtype=torch.float32).npu()
    offset = None
    
    expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale = torch_npu.npu_moe_init_routing_v2(
                    x, expert_idx, scale=scale, offset=offset,
                    active_num=active_num, expert_capacity=expert_capacity, expert_num=expert_num, drop_pad_mode=drop_pad_mode, 
                    expert_tokens_num_type=expert_tokens_num_type, expert_tokens_num_flag=expert_tokens_num_flag,
                    active_expert_range=active_expert_range, quant_mode=quant_mode, row_idx_type=row_idx_type)
    ```

-   图模式调用

    ```python
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    
    class MoeInitRoutingV2Model(nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, x, expert_idx, *, scale=None, offset=None, active_num=-1, expert_capacity=-1,
                    expert_num=-1, drop_pad_mode=0, expert_tokens_num_type=0, expert_tokens_num_flag=False,
                    quant_mode=0, active_expert_range=0, row_idx_type=0):
            return torch.ops.npu.npu_moe_init_routing_v2(x, expert_idx, scale=scale, offset=offset,
                    active_num=active_num, expert_capacity=expert_capacity, expert_num=expert_num, drop_pad_mode=drop_pad_mode, 
                    expert_tokens_num_type=expert_tokens_num_type, expert_tokens_num_flag=expert_tokens_num_flag,
                    active_expert_range=active_expert_range, quant_mode=quant_mode, row_idx_type=row_idx_type)
    
    def main():
        bs = 1
        h = 613
        k = 475
    
        active_num = 475
        expert_capacity = -1
        expert_num = 226
        drop_pad_mode = 0
        expert_tokens_num_type = 1
        expert_tokens_num_flag = True
        quant_mode = -1
        active_expert_range = [23, 35]
        row_idx_type = 0
    
        x = torch.randn((bs, h), dtype=torch.float32).npu()
        expert_idx = torch.randint(0, expert_num, (bs, k), dtype=torch.int32).npu()
        scale = torch.randn((bs,), dtype=torch.float32).npu()
        offset = None
    
        moe_init_routing_v2_model = MoeInitRoutingV2Model().npu()
        moe_init_routing_v2_model = torch.compile(moe_init_routing_v2_model, backend=npu_backend, dynamic=False)
        expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale = moe_init_routing_v2_model(x,
                                        expert_idx, scale=scale, offset=offset, active_num=active_num,
                                        expert_capacity=expert_capacity, expert_num=expert_num, drop_pad_mode=drop_pad_mode, 
                                        expert_tokens_num_type=expert_tokens_num_type, expert_tokens_num_flag=expert_tokens_num_flag,
                                        active_expert_range=active_expert_range, quant_mode=quant_mode, row_idx_type=row_idx_type)
    
    if __name__ == '__main__':
        main()
    ```


