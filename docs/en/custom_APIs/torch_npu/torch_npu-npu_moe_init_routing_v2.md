# torch_npu.npu_moe_init_routing_v2<a name="en-us_topic_0000002309015148"></a>

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |
|<term>Atlas inference products</term> | √   |

## Function<a name="en-us_topic_0000002271534921_section1650913464367"></a>

- Description: Performs mixture of experts (MoE) routing based on the computation results of [torch_npu.npu_moe_gating_top_k_softmax](torch_npu-npu_moe_gating_top_k_softmax.md). Non-quantization, dynamic quantization, and static quantization configurations are supported.
- Formulas: 

    1. Sort the input `expertIdx` to obtain the sorted expert indices `sortedExpertIdx` and the corresponding row indices `sortedRowIdx`.

        $$
        sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
        $$

    2. Perform position mapping using `sortedRowIdx` to obtain `expandedRowIdxOut`.
        - When `rowIdxType` is `1`, the API outputs scatter indices.
        $$
        expandedRowIdxOut[i]=sortedRowIdx[i]
        $$
        - When `rowIdxType` is `0`, the API outputs gather indices.
        $$
        expandedRowIdxOut[sortedRowIdx[i]]=i
        $$
      
    3. Compute the histogram of `sortedExpertIdx` for each expert to obtain `expertTokensCountOrCumsumOutOptional`.

        $$
        expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
        $$

    4. When `quantMode` is not `-1`, compute the quantization results.
        - Static quantization:
        $$
        quantResult=round((x∗scaleOptional)+offsetOptional)
        $$
        
        - Dynamic quantization:
            - If `scale` is not specified:
                $$
                dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
                $$

                $$
                quantResult = round(x / dynamicQuantScaleOutOptional)
                $$
            - If `scale` is specified:
                $$
                dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
                $$

                $$
                quantResult = round(x / dynamicQuantScaleOutOptional)
                $$
  
    5. Tokens are rearranged using scatter indices when the active expert range covers all experts. In other configurations, tokens are rearranged using gather indices. When `dropPadMode` is set to `1`, the number of tokens processed by each expert is padded to `expertCapacity`. Tokens exceeding `expertCapacity` are dropped, and insufficient tokens are padded with zeros. The resulting output `expandedXOut` is obtained as follows:
        - Non-quantization scenarios:
        - Rearrangement using scatter indices:
        $$
        expandedXOut[i]=x[scatterRowIdx[i]//K]
        $$
        - Rearrangement using gather indices:
        $$
        expandedXOut[gatherRowIdx[i]]=x[i//K]
        $$
        - Quantization scenarios:
        - Rearrangement using scatter indices:
        $$
        expandedXOut[i]=quantResult[scatterRowIdx[i]//K]
        $$
        - Rearrangement using gather indices:
        $$
        expandedXOut[gatherRowIdx[i]]=quantResult[i//K]
        $$

    6. The valid element count of `expandedRowIdxOut`, denoted by `availableIdxNum`, is calculated as the number of elements in `expertIdx` that fall within the range specified by `activeExpertRangeOptional`.
        $$
        availableIdxNum = |\{x\in expertIdx| expert\_start \le x<expert\_end \ \}|
        $$

- Equivalent computation logic:

    ```python
    import numpy as np
    import random


    def simplified_mx_quantize(fp_array: np.ndarray, mx_ele_dtype: str = "float8_e4m3fn") -> tuple:
        """
        Simplified MX quantization function
        Input: fp_array (shape=[n, h], dtype=fp16/bf16)
        Output: (scale_array, ele_array)
        """
        try:
            from ml_dtypes import float8_e5m2, float8_e4m3fn
            from en_dtypes import float8_e8m0
        except ImportError:
            raise AssertionError("Unsupported UT testcase due to lack of package ml_dtypes or en_dtypes")

        # --- 1. Parameters and constants---
        BLOCK_SIZE = 32
        AXIS = -1 # Always process the last dimension

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

        # --- 2. Padding and reshaping (blocking)---
        # Convert [N, H] to [N, H_blocks, 32]
        orig_shape = fp_array.shape
        h_dim = orig_shape[AXIS]

        # Calculate the required padding length
        pad_len = (BLOCK_SIZE - (h_dim % BLOCK_SIZE)) % BLOCK_SIZE
        if pad_len > 0:
            # Pad zeros only along the last dimension
            pad_width = [(0, 0)] * fp_array.ndim
            pad_width[AXIS] = (0, pad_len)
            fp_array = np.pad(fp_array, pad_width, 'constant')

        padded_shape = fp_array.shape
        # Reshape to (..., blocks, block_size)
        new_shape = list(padded_shape)
        new_shape[AXIS] = new_shape[AXIS] // BLOCK_SIZE
        new_shape.append(BLOCK_SIZE)
        fp_array_blocked = fp_array.reshape(new_shape)

        # --- 3. Compute the shared scale (shared exponent)---
        # Logic: scale = floor(log2(max(abs(block)))) - ele_emax
        ele_emax = int(np.log2(max_norm))
        # Find the maximum absolute value within each block
        fp_abs_max = np.max(np.abs(fp_array_blocked), axis=-1, keepdims=True)

        # Avoid log2(0)
        FP32_MIN_NORMAL = 2 ** (-126)
        share_exp = np.floor(
            np.log2(fp_abs_max + FP32_MIN_NORMAL * (fp_abs_max == 0))) - ele_emax

        # Handle special values and clamp to the E8M0 range
        share_exp[fp_abs_max == 0] = -float("inf")
        SCALE_EMAX = 127
        share_exp[share_exp > SCALE_EMAX] = float("NaN")
        share_exp[share_exp < -SCALE_EMAX] = -SCALE_EMAX

        # --- 4. Quantize elements---
        # Formula: scaled = input / 2^scale
        scale_val = 2.0 ** share_exp
        scaled_input = fp_array_blocked / scale_val

        # Simulate FP8 precision loss through rounding and clamping
        # Compute the private exponent
        min_exp = -(2 ** (exp_bits - 1)) + 2
        abs_scaled = np.abs(scaled_input)
        # Avoid log2(0)
        private_exp = np.floor(
            np.log2(abs_scaled + (abs_scaled == 0))).astype(np.int32)
        # private_exp = np.clip(private_exp, min=min_exp, max=None)
        private_exp = np.clip(private_exp, min_exp, None)

        # Round the mantissa by using round-to-nearest-even (rint)
        step_scale = 2.0 ** (mantissa_bits - private_exp)
        ret = scaled_input * step_scale
        ret = np.rint(ret)
        ret = ret / step_scale

        # Clamp elements to the maximum norm
        ret = np.clip(ret, -max_norm, max_norm)

        # --- 5. Restore shape and convert format---
        # Restore the shape to [N, H_padded]
        ele_array = ret.reshape(padded_shape)
        # Remove the padded dimensions
        if pad_len > 0:
            ele_array = ele_array[..., :h_dim]

        # Process the shape of the scale arrays
        # Remove the trailing dimension from shape (N, H_blocks, 1)
        share_exp = np.squeeze(share_exp, axis=-1)
        scale_array = 2.0 ** share_exp

        # Align the scale array to an even length to meet Cube hardware requirements
        if scale_array.shape[-1] % 2 != 0:
            scale_array = np.pad(scale_array, ((0, 0), (0, 1)),
                                'constant', constant_values=2**-127)

        # Reshape the scale array to (N, H_blocks/2, 2)
        s_shape = list(scale_array.shape)
        s_shape[-1] //= 2
        s_shape.append(2)
        scale_array = scale_array.reshape(s_shape)

        # --- 6. Final type conversion ---
        # Convert the scale array to E8M0
        if float8_e8m0:
            scale_array = scale_array.astype(float8_e8m0)

        # Convert the element array to the target FP8 type
        # Convert to float32 first to ensure compatibility, especially for bf16 inputs
        ele_array = ele_array.astype(np.float32)
        ele_array = np.nan_to_num(ele_array, nan=0.0)
        if target_dtype:
            ele_array = ele_array.astype(target_dtype)

        return scale_array, ele_array


    class MoeInitRoutingV2CPU:
        """
        Equivalent implementation of MoeInitRoutingV2 on the CPU
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
            Equivalent compute logic of MoeInitRoutingV2 on the CPU
            
            Parameters:
                x: input tensor, with shape (bs, h).
                expert_idx: expert index, with shape (bs, k).
                scale: Optional. Quantization scale.
                offset: Optional. Quantization offset.
                expert_range: expert range [start, end]. The default value is [0, 16].
                quant_mode: quantization mode. Valid values: -1 (no quantization), 0 (static quantization), 1 (dynamic quantization), or 2/3 (MXFP8)
                row_idx_type: row_idx type (valid values: 0 or 1)
                expert_tokens_num_flag: specifies whether to calculate the number of expert tokens
                expert_tokens_num_type: number of expert tokens (valid values: 0, 1, or 2)
                drop_pad_mode: drop_pad mode (valid values: 0 or 1)
                active_num: number of activations
                expert_capacity: expert capacity
                
            Returns:
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
            Generate test input data
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
        """Non-quantization mode"""
        print("=" * 50)
        print("Demo: Non-quantization mode (quant_mode=-1)")
        print("=" * 50)
        
        bs, h, k = 32, 200, 5
        expert_range = [0, 16]
        
        # Generate the inputs
        x, expert_idx, scale, offset, expert_tokens_num_type, row_idx_type, active_num, expert_capacity = \
            MoeInitRoutingV2CPU.generate_inputs(bs, h, k, np.float16, (bs,), False, True, 0)
        
        # Execute computation
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
        """Static quantization mode"""
        print("=" * 50)
        print("Demo: Static quantization mode (quant_mode=0)")
        print("=" * 50)
        
        bs, h, k = 32, 200, 5
        expert_range = [0, 16]
        
        # Generate the inputs (scale and offset are required)
        x, expert_idx, scale, offset, expert_tokens_num_type, row_idx_type, active_num, expert_capacity = \
            MoeInitRoutingV2CPU.generate_inputs(bs, h, k, np.float32, (1,), False, False, 0)
        
        # Execute computation
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
        """Dynamic quantization mode"""
        print("=" * 50)
        print("Demo: Dynamic quantization mode (quant_mode=1)")
        print("=" * 50)
        
        bs, h, k = 32, 200, 8
        expert_range = [0, 16]
        expert_range_length = expert_range[1] - expert_range[0]
        
        # Generate the inputs (scale is optional)
        x, expert_idx, scale, offset, expert_tokens_num_type, row_idx_type, active_num, expert_capacity = \
            MoeInitRoutingV2CPU.generate_inputs(bs, h, k, np.float32, (expert_range_length, h), False, True, 0)
        
        # Execute computation
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
        """drop_pad mode"""
        print("=" * 50)
        print("Demo: drop_pad_mode=1")
        print("=" * 50)
        
        bs, h, k = 32, 200, 5
        expert_range = [0, 32]  # The full range is usually used when drop_pad_mode is set to 1
        
        # Generate the inputs
        x, expert_idx, scale, offset, expert_tokens_num_type, row_idx_type, active_num, expert_capacity = \
            MoeInitRoutingV2CPU.generate_inputs(bs, h, k, np.float16, (bs,), False, True, 1)
        
        # Execute computation
        expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale = \
            MoeInitRoutingV2CPU.cpu_op_exec(
                x, expert_idx, scale, offset,
                expert_range=expert_range,
                quant_mode=-1,
                row_idx_type=0, # When drop_pad_mode is set to 1, row_idx_type is forcibly set to 0.
                expert_tokens_num_flag=True,
                expert_tokens_num_type=1,  # The value is forcibly set to 1 when expert_tokens_num_type=1
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

## Prototype<a name="en-us_topic_0000002271534921_section14509346133618"></a>

```python
torch_npu.npu_moe_init_routing_v2(x, expert_idx, *, scale=None, offset=None, active_num=-1, expert_capacity=-1, expert_num=-1, drop_pad_mode=0, expert_tokens_num_type=0, expert_tokens_num_flag=False, quant_mode=-1, active_expert_range=[], row_idx_type=0) -> (Tensor, Tensor, Tensor, Tensor)
```

## Parameters<a name="en-us_topic_0000002271534921_section2050919466367"></a>

- **`x`** (`Tensor`): Required. Input token features for MoE. This parameter must be 2D with shape `(NUM_ROWS, H)`. The data type can be `float16`, `bfloat16`, `float32`, or `int8`. The data layout must be ND.
- **`expert_idx`** (`Tensor`): Required. Selected K processing experts corresponding to each row feature in the output of [torch_npu.npu_moe_gating_top_k_softmax](torch_npu-npu_moe_gating_top_k_softmax.md). This parameter must be 2D with shape `(NUM_ROWS, K)`. The expert ID must be less than or equal to the expert count. The data type is `int32`. The data layout must be ND.
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`scale`** (`Tensor`): Optional. Parameter used to compute the quantization results. This parameter can be set to `None`. The data type is `float32`. The data layout must be ND. If this parameter is omitted, it indicates that `scale` is not used during computation, and the values in the output `expanded_scale` are meaningless.
    - If provided in non-quantization scenarios, this parameter must be 1D with shape `(NUM_ROWS,)`.
    - In static quantization scenarios, this parameter must be provided. It must be 1D with shape `(1,)`.
    - If provided in dynamic quantization scenarios, this parameter must be 2D with shape `(expert_end - expert_start, H)` or `(1, H)`.

- **`offset`** (`Tensor`): Optional. Offset value used to compute the quantization results. This parameter can be set to `None`. The data type is `float32`. The data layout must be ND.
    - In non-quantization scenarios, this parameter is omitted.
    - In static quantization scenarios, this parameter must be provided. It must be 1D with shape `(1,)`.
    - In dynamic quantization scenarios, this parameter is omitted.

- **`active_num`** (`int`): Optional. The maximum number of rows to be processed. The default value is `-1`. Only the first `active_num` rows in `expanded_x` are valid. The input must be greater than or equal to `0`, where `0` indicates a dropless configuration and values greater than `0` indicate an active configuration, which constrains the total number of tokens processed by all experts.
- **`expert_capacity`** (`int`): Optional. Number of tokens each expert can process. The default value is `-1`. Input parameter verification requires the value to be greater than 0 and less than `NUM_ROWS`.
- **`expert_num`** (`int`): Optional. Expert count. The default value is `-1`. When `expert_tokens_num_type` is set to `key_value` mode, the value range is [0, 5120]. In other modes, the value range is [0, 10240].
- **`drop_pad_mode`** (`int`): Optional. Specifies whether to enable drop_pad mode. The default value is `0`. 0: enables dropless mode, where `expert_capacity` is not verified. 1: enables drop_pad mode.
- **`expert_tokens_num_type`** (`int`): Optional. Histogram mode. The default value is `0`. Valid values are `0`, `1`, or `2`. `0` enables cumsum mode; `1` enables count mode indicating that the output value is the cumulative number of tokens processed by each expert; and `2` enables `key_value` mode indicating that the output value contains the expert IDs and the corresponding cumulative numbers of tokens processed by each expert.
- **`expert_tokens_num_flag`** (`bool`): Optional. Specifies whether to output `expert_token_cumsum_or_count`. Valid values are `False` or `True`. The default value is `False`.
- **`quant_mode`** (`int`): Optional. Quantization mode. The default value is `-1`. Valid values are `0`, `1`, or `-1`. The value `0` indicates static quantization, `-1` indicates non-quantization, and `1` indicates dynamic quantization.
- **`active_expert_range`** (`List[int]`): Optional. Range of active experts. The default value is an empty list. The value range is [expert_start, expert_end), which is left-closed and right-open, indicating that active experts range from expert_start to expert_end. The values must be greater than or equal to 0, and `expert_end` must be less than or equal to `expert_num`. In drop_pad scenarios, `expert_start` is 0 and `expert_end` is equal to `expert_num`. When the default value is provided, the range of active experts is considered to be between 0 and `expert_num`.
- **`row_idx_type`** (`int`): Optional. Index type used for the output `expanded_row_idx`. The default value is `0`. Valid values are `0` or `1`. The value `0` enables gather indices, and the value `1` enables scatter indices.

## Return Values<a name="en-us_topic_0000002271534921_section18510124618368"></a>

- **`expanded_x`** (`Tensor`): Features extended based on `expert_idx`. In dropless scenarios, the shape is `(NUM_ROWS * K, H)`. In active scenarios, the shape is `(min(activeNum, NUM_ROWS * K), H)`. In drop-pad scenarios, this parameter must be 3D with shape `(expertNum, expertCapacity, H)`. The data type must be identical to that of `x` in non-quantization scenarios. The data type is `int8` in quantization scenarios. The data layout must be ND. In quantization configurations, the output value is meaningless when the data type of `x` is `int8`.
- **`expanded_row_idx`** (`Tensor`): Mapping between `expanded_x` and `x`. This parameter must be 1D with shape `(NUM_ROWS * K,)`. The data type is `int32`. The data layout must be ND. When `row_idx_type` is `1`, the first `available_idx_num` elements are valid data. Invalid data is uninitialized. When `row_idx_type` is `0`, invalid elements are padded with `-1`.
- **`expert_token_cumsum_or_count`** (`Tensor`): Statistical results or cumulative values of the numbers of tokens processed by each expert.
    - When `expert_tokens_num_type` is `0`, this parameter indicates the prefix sums of the numbers of tokens processed by sorted experts within the `active_expert_range`.
    - When `expert_tokens_num_type` is `1`, this parameter must be 1D with shape `(expert_end - expert_start,)`, indicating the total numbers of tokens processed by experts within the `active_expert_range`.
    - When `expert_tokens_num_type` is `2`, this parameter must be 2D with shape `(expert_num, 2)`, indicating the experts with non-zero token counts within the `active_expert_range` and the corresponding total numbers of tokens processed by each expert.

    Element pairs where `expert_idx` falls within the `active_expert_range` and elements with a token count of `0` are excluded represent valid element pairs. These valid pairs are stored at the beginning of the tensor while preserving the original order. The data type is `int64`. The data layout must be ND.
- **`expanded_scale`** (`Tensor`): The data type is `float32`. The data layout must be ND. The output shape is the product of all dimensions of `expert_idx` except the last dimension. Let `available_idx_num` represent the number of elements within the `active_expert_range`.
    - Atlas A2 training products/Atlas A2 inference products/Atlas A3 training products/Atlas A3 inference products:
        - In non-quantization scenarios, the first `available_idx_num` elements are valid data when `scale` is provided.
        - In dynamic quantization scenarios, the output contains intermediate values of `scale` from the quantization computation process, and the first `available_idx_num` elements are valid data.
        - This output is not available in static quantization scenarios.
    - Atlas inference products: This output is replaced by `expert_tokens_before_capacity`, which must be a tensor with shape `(expert_num,)`, indicating the statistical results of the numbers of tokens processed by each expert before dropping.

## Constraints<a name="en-us_topic_0000002271534921_section75102046193618"></a>

Atlas A2 training products/Atlas A2 inference products/Atlas A3 training products/Atlas A3 inference products:

- This API can be used in inference scenarios.
- This API supports graph mode.
- All the following conditions must be satisfied to enable the low-latency performance template.
    - The input shape requirements for `x`, `expert_idx`, and `scale` must be `(1, 7168)`, `(1, 8)`, and `(256, 7168)`, respectively.
    - The data type of `x` must be `bfloat16`.
    - Attribute requirements: `active_expert_range=[0, 256]`, `quant_mode = 1`, `expert_tokens_num_type=2`, and `expert_num=256`.
- All the following conditions must be satisfied to enable the large-batch performance template.
    - The value range of `NUM_ROWS` is [384, 8192].
    - K=8
    - expert_num=256
    - expert_end-expert_start<=32
    - quant_mode=-1
    - row_idx_type=1
    - expert_tokens_num_type=1
- In scenarios where the operator input shape is small, the multi-core synchronization time between operations accounts for a high proportion and becomes a performance bottleneck. Therefore, the full-load performance template is added for this specialized scenario. In this template, data transfer, sorting, and computation are all completed within a single kernel. The condition `drop_pad_mode=0` must be satisfied.

Special constraints for Atlas inference products:

- **Input `x` data type**: can only be `float16` or `float32`. The data type `bfloat16` is not supported.
- **Quantization mode**: Only non-quantization scenarios where `quant_mode = -1` are supported. Static quantization where `quant_mode = 0`, dynamic quantization where `quant_mode = 1`, and quantization modes such as MXFP8 or HIF8 are not supported.
- **drop_pad mode**: Only the dropless scenario (`drop_pad_mode=0`) is supported. The drop_pad scenario (`drop_pad_mode=1`) is not supported. The input parameter `drop_pad_mode` is forcibly set to `0` regardless of its value at runtime.
- **`expert_capacity` parameter**: At runtime, the input parameter `expert_capacity` is forcibly set to `0` regardless of its value.
- **`expert_tokens_num_type`**: The input parameter can be set to `0`, `1`, or `2`. In this scenario, the underlying computation is executed strictly in cumsum mode (prefix sum) regardless of the input value.
- **Output parameter differences**: The fourth return value is `expert_tokens_before_capacity`, which must be a tensor with shape `(expert_num,)`, indicating the statistical results of the numbers of tokens processed by each expert before dropping.

## Examples<a name="en-us_topic_0000002271534921_section12510194643618"></a>

- Single-operator call

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

- Graph mode call

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
