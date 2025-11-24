import os
import copy
import shutil
import random
import torch
import torch_npu
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device, SupportedDevices


def cumsum2index(seq_lens):
    if seq_lens is None:
        raise KeyError('seq_lens is None')
    seq_lens = np.asarray(seq_lens)
    batch = len(seq_lens) - 1
    result = np.zeros(batch, dtype=seq_lens.dtype)
    for i in range(batch):
        result[i] = int(seq_lens[i + 1] - seq_lens[i])
    return result


def gather_pa_kv_cache_nd(context, key_cache, value_cache, block_tables, seq_lens, key, value, seq_offset):
    num_blocks, block_size, num_heads, head_size_k = key_cache.shape
    is_seq_lens_cumsum = context.get('is_seq_lens_cumsum', False)
    num_tokens = key.shape[0]
    kv_rslt_id = 0

    if is_seq_lens_cumsum:
        seq_lens = cumsum2index(seq_lens)

    accum_seq_len = 0
    for i in range(len(seq_lens)):
        block_table = block_tables[i]
        seq_len = seq_lens[i]

        if num_blocks > accum_seq_len and num_tokens <= (acc_seq_len + seq_len):
            seq_len = num_tokens - accum_seq_len
        accum_seq_len += seq_len

        if seq_offset is None:
            block_start = 0
        else:
            block_start = seq_offset[i] // block_size
        
        for j in range(seq_len):
            if kv_rslt_id >= key.shape[0]:
                break
            
            block_table_idx = block_start + j // block_size
            if block_table_idx >= block_table.shape[0]:
                is_filled_with_zero = True
                block_id = -1
            else:
                is_filled_with_zero = False
                block_id = block_table[block_table_idx]
            
            block_offset = j % block_size

            if block_id >= num_blocks or block_id < 0 or is_filled_with_zero:
                temp_k = np.zeros_like(key_cache[0][0])
                temp_v = np.zeros_like(value_cache[0][0])
            else:
                temp_k = key_cache[block_id][block_offset]
                temp_v = value_cache[block_id][block_offset]

            key[kv_rslt_id] = temp_k
            value[kv_rslt_id] = temp_v
            kv_rslt_id += 1

    return [key, value]


def gather_pa_kv_cache_nz(context, key_cache, value_cache, block_tables, seq_lens, key, value, seq_offset):
    num_blocks, _, block_size, elenum_aligned = key_cache.shape
    num_tokens, num_heads, head_size_k = key.shape
    num_tokens, num_heads, head_size_v = value.shape
    is_seq_lens_cumsum = context.get('is_seq_lens_cumsum', False)

    num_heads_k = num_heads * head_size_k
    num_heads_v = num_heads * head_size_v

    key = key.reshape((num_tokens, num_heads_k))
    value = value.reshape((num_tokens, num_heads_v))

    if is_seq_lens_cumsum:
        seq_lens = cumsum2index(seq_lens)

    kv_rslt_id = 0

    for i in range(len(seq_lens)):
        block_table = block_tables[i]
        seq_len = seq_lens[i]

        if seq_offset is None:
            block_table = 0
        else:
            block_start = seq_offset[i] // block_size
        
        for j in range(seq_len):
            if kv_rslt_id >= key.shape[0]:
                break

            block_table_idx = block_start + j // block_size
            if block_table_idx >= block_table.shape[0]:
                block_id = -1
            else:
                block_id = block_table[block_table_idx]
                block_offset = j % block_size
            
            temp_k = np.zeros_like((num_heads_k,), dtype=key.dtype)
            temp_v = np.zeros_like((num_heads_v,), dtype=value.dtype)

            if block_id >= 0 and block_id < num_blocks:
                for k in range(num_heads_k // elenum_aligned):
                    temp_k[k * elenum_aligned: (k + 1) * elenum_aligned] = \
                        key_cache[block_id][k][block_offset][:]
                for k in range(num_heads_v // elenum_aligned):
                    temp_v[k * elenum_aligned: (k + 1) * elenum_aligned] = \
                    value_cache[block_id][k][block_offset][:]

            key[kv_rslt_id] = temp_k
            value[kv_rslt_id] = temp_v
            kv_rslt_id += 1
    
    key = key.reshape((num_tokens, num_heads, head_size_k))
    value = value.reshape((num_tokens, num_heads, head_size_v))
    return [key, value]
    

def golden_gather_pa_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    key_out: torch.Tensor,
    value_out: torch.Tensor,
    seq_offset: torch.Tensor = None,
    is_seq_lens_cumsum: bool = False,
    cache_mode: str = 'Norm'
):

    key_cache_np = key_cache.cpu().numpy()
    value_cache_np = value_cache.cpu().numpy()
    block_tables_np = block_tables.cpu().numpy()
    seq_lens_np = seq_lens.cpu().numpy()
    key_np = key.cpu().numpy().copy()
    value_np = value.cpu().numpy().copy()
    seq_offset_np = seq_offset.cpu().numpy() if seq_offset is not None else None

    context = {
        'is_seq_lens_cumsum': is_seq_lens_cumsum,
        'cache_mode': cache_mode
    }

    if cache_mode == 'Norm':
        key_result, value_result = gather_pa_kv_cache_nd(
            context=context,
            key_cache=key_cache_np,
            value_cache=value_cache_np,
            block_tables=block_tables_np,
            seq_lens=seq_lens_np,
            key=key_np,
            value=value_np,
            seq_offset=seq_offset_np
        )
    elif cache_mode == 'PA_NZ':
        key_result, value_result = gather_pa_kv_cache_nz(
            context=context,
            key_cache=key_cache_np,
            value_cache=value_cache_np,
            block_tables=block_tables_np,
            seq_lens=seq_lens_np,
            key=key_np,
            value=value_np,
            seq_offset=seq_offset_np
        )
    else:
        raise KeyError(f'cache mode can only be one of Norm or PA_NZ')

    return torch.from_numpy(key_result), torch.from_numpy(value_result)


class GatherPaKvCacheModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, key_cache, value_cache, block_tables, seq_lens, key, value, seq_offset, is_seq_lens_cumsum):
        output_key, output_value = torch_npu.npu_gather_pa_kv_cache_functional(
            key_cache, value_cache, block_tables, seq_lens, key, value, 
            seq_offset=seq_offset, is_seq_lens_cumsum=is_seq_lens_cumsum
        )
        return output_key, output_value


class GatherPaKvCacheInplaceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, key_cache, value_cache, block_tables, seq_lens, key, value, seq_offset, is_seq_lens_cumsum):
        torch_npu.npu_gather_pa_kv_cache(
            key_cache, value_cache, block_tables, seq_lens, key, value, 
            seq_offset=seq_offset, is_seq_lens_cumsum=is_seq_lens_cumsum
        )
        return key, value


class TestGatherPaKvCache(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        random.seed(0)
        torch.manual_seed(0)
    
    def _run_test(self, mode, api_impl_mode, input_dtype=torch.float16):
        batch_size = 2
        num_blocks = 8
        head_num = 4
        block_size = 64
        head_dim = 64
        max_blocks_per_sequence = 5

        seq_lens_list = [random.randint(1, 10) for _ in range(batch_size)]
        is_seq_lens_cumsum = True

        if is_seq_lens_cumsum:
            cumsum = [0]
            for x in seq_lens_list:
                cumsum.append(cumsum[-1] + x)
            seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
            total_tokens = cumsum[-1]
        else:
            seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
            total_tokens = sum(seq_lens_list)

        key_cache = torch.randn(num_blocks, block_size, head_num, head_dim, dtype=input_dtype)
        value_cache = torch.randn(num_blocks, block_size, head_num, head_dim, dtype=input_dtype)
        block_tables = torch.randint(0, num_blocks, (batch_size, max_blocks_per_sequence), dtype=torch.int32)
        key_out = torch.zeros(total_tokens, head_num, head_dim, dtype=input_dtype)
        value_out = torch.zeros(total_tokens, head_num, head_dim, dtype=input_dtype)
        seq_offset = torch.randint(0, block_size, (batch_size,), dtype=torch.int32)

        key_gold, value_gold = golden_gather_pa_kv_cache(
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            key_out=key_out,
            value_out=value_out,
            seq_offset=seq_offset,
            is_seq_lens_cumsum=is_seq_lens_cumsum,
            cache_mode='Norm'
        )

        key_cache_npu = key_cache.npu()
        value_cache_npu = value_cache.npu()
        block_tables_npu = block_tables.npu()
        seq_lens_npu = seq_lens.npu()
        key_out_npu = key_out.npu()
        value_out_npu = value_out.npu()
        seq_offset_npu = seq_offset.npu()

        if mode == "inplace":
            model = GatherPaKvCacheInplaceModel()
            key_input_npu = key_out_npu
            value_input_npu = value_out_npu
        elif mode == "out_of_place":
            model = GatherPaKvCacheModel()
            key_input_npu = key_out_npu
            value_input_npu = value_out_npu
        else:
            self.fail(f"Unsupported mode: {mode}")

        if api_impl_mode == "eager":
            model = torch.compile(model, backend="eager", dynamic=True)
        else:
            self.fail(f"Unsupported api_impl_mode: {api_impl_mode}")

        with torch.npu_gard():
            out_key_npu, out_value_npu = model(
                key_cache_npu, value_cache_npu, block_tables_npu, seq_lens_npu,
                key_input_npu, value_input_npu,
                seq_offset=seq_offset_npu,
                is_seq_lens_cumsum=is_seq_lens_cumsum
            )
        
        torch.npu.synchronize()
        out_key_cpu = out_key_npu.cpu()
        out_value_cpu = out_value_npu.cpu()

        rtol = 1e-3 if input_dtype == torch.float16 else 1e-4
        atol = 1e-3 if input_dtype == torch.float16 else 1e-4

        self.assertRtolEqual(key_gold, out_key_cpu, rtol=rtol, atol=atol)
        self.assertRtolEqual(value_gold, out_value_cpu, rtol=rtol, atol=atol)

    @SupportedDevices(['Ascend910_95'])
    def test_out_of_place_eager_fp16(self):
        self._run_test("out_of_place", "eager", torch.float16)

    @SupportedDevices(['Ascend910_95'])
    def test_inplace_eager_fp16(self):
        self._run_test("inplace", "eager", torch.float16)


if __name__ == "__main__":
    run_tests()