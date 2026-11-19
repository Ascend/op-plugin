# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# Licensed under the BSD 3-Clause License. See LICENSE for details.

import itertools
import math
import unittest

import numpy as np
import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


FP4_VALUES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
FP8_VALUES = np.array([
    code * 2.0 ** -9 if code < 8 else
    (1 + (code & 7) / 8) * 2.0 ** ((code >> 3) - 7)
    for code in range(127)
], dtype=np.float32)


def encode_values(values, table, mode="rint"):
    """Enumerate adjacent representable values without using a device cast."""
    magnitude = np.minimum(np.abs(values), table[-1])
    high = np.minimum(np.searchsorted(table, magnitude), len(table) - 1)
    low = np.maximum(high - 1, 0)
    if mode == "floor":
        below = np.maximum(np.searchsorted(table, magnitude, side="right") - 1, 0)
        codes = np.where(np.signbit(values), high, below)
    else:
        lower_distance = magnitude.astype(np.float64) - table[low]
        upper_distance = table[high].astype(np.float64) - magnitude
        upper_even = True if mode == "round" else high % 2 == 0
        codes = np.where(
            (upper_distance < lower_distance) |
            ((upper_distance == lower_distance) & upper_even), high, low)
    sign_bit = 8 if len(table) == 8 else 128
    codes = codes.astype(np.uint8) | np.where(np.signbit(values), sign_bit, 0).astype(np.uint8)
    return np.where(np.isnan(values), 7 if len(table) == 8 else 127, codes).astype(np.uint8)


def nvfp4_reference(source, mode="rint", max_low_bound=0.0):
    """CPU oracle: source-dtype abs/max, FP32 scale arithmetic, raw packed bytes."""
    if source.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("NVFP4 reference requires FP16 or BF16 source values")
    k = source.shape[-1]
    rows = math.prod(source.shape[:-1])
    groups = (k + 15) // 16
    scale_shape = (*source.shape[:-1], (groups + 1) // 2, 2)
    scales = np.zeros((rows, (groups + 1) // 2 * 2), dtype=np.uint8)
    if source.numel() == 0:
        return (torch.empty((*source.shape[:-1], k // 2), dtype=torch.uint8),
                torch.from_numpy(scales.reshape(scale_shape)))

    padded = torch.full((rows, groups * 16), float("nan"), dtype=source.dtype)
    padded[:, :k] = source.reshape(rows, k)
    blocks = padded.reshape(rows, groups, 16)
    all_nan = torch.isnan(blocks).all(dim=-1).numpy()
    # The maximum is selected in the original FP16/BF16 dtype before widening.
    amax = torch.where(torch.isnan(blocks), 0, blocks.abs()).amax(dim=-1).float().numpy()
    part = blocks.float().numpy()
    with np.errstate(all="ignore"):
        scale32 = np.maximum(amax, np.float32(max_low_bound)) * np.float32(float.fromhex("0x1.555556p-3"))
        scale_codes = encode_values(np.minimum(scale32, np.float32(448)), FP8_VALUES)
        scale_codes[amax == 0] = 0
        scale_codes[all_nan] = 127
        inverse = np.divide(np.float32(1), FP8_VALUES[np.minimum(scale_codes, 126)])
        inverse[amax == 0] = 0
        product = part * inverse[..., None]
        codes = encode_values(product, FP4_VALUES, mode).reshape(rows, groups * 16)[:, :k]
    scales[:, :groups] = scale_codes
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    return (torch.from_numpy(packed.reshape(*source.shape[:-1], k // 2)),
            torch.from_numpy(scales.reshape(scale_shape)))


class TestDynamicMxQuantMeta(unittest.TestCase):
    def test_nvfp4_shapes(self):
        for dtype, rank, k in itertools.product((torch.float16, torch.bfloat16), range(1, 8), (0, 2, 18, 34)):
            shape = (2,) * (rank - 1) + (k,)
            with self.subTest(dtype=dtype, shape=shape):
                x = torch.empty(shape, dtype=dtype, device="meta")
                y, scale = torch_npu.npu_dynamic_mx_quant(x, scale_alg=3, block_size=16)
                self.assertEqual(tuple(y.shape), (*shape[:-1], k // 2))
                self.assertEqual(tuple(scale.shape), (*shape[:-1], ((k + 15) // 16 + 1) // 2, 2))
                self.assertEqual(y.dtype, torch.uint8)
                self.assertEqual(scale.dtype, torch.uint8)

    def test_nvfp4_invalid_arguments(self):
        x = torch.empty((2, 18), dtype=torch.float16, device="meta")
        invalid = (
            {"block_size": 32}, {"axis": 0}, {"dst_type": 297},
            {"dst_type_max": 6.0}, {"max_low_bound": -1.0},
            {"max_low_bound": float("inf")}, {"max_low_bound": float("nan")},
            {"max_low_bound": 1e40}, {"round_mode": "ceil"},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(RuntimeError, "NVFP4"):
                kwargs = {"scale_alg": 3, "block_size": 16, **overrides}
                torch_npu.npu_dynamic_mx_quant(x, **kwargs)
        for shape, dtype in (((2, 18), torch.float32), ((2,) * 7 + (16,), torch.float16),
                             ((2, 17), torch.float16)):
            with self.subTest(shape=shape, dtype=dtype), self.assertRaises(RuntimeError):
                torch_npu.npu_dynamic_mx_quant(
                    torch.empty(shape, dtype=dtype, device="meta"), scale_alg=3, block_size=16)
        for shape in ((), (2,) * 7 + (16,)):
            with self.subTest(shape=shape), self.assertRaisesRegex(RuntimeError, r"NVFP4 requires input rank 1\.\.7"):
                torch_npu.npu_dynamic_mx_quant(
                    torch.empty(shape, dtype=torch.float16, device="meta"), scale_alg=3, block_size=16)
        with self.assertRaisesRegex(RuntimeError, "NVFP4 requires the last axis"):
            torch_npu.npu_dynamic_mx_quant(x, scale_alg=3, block_size=16, axis=0)

    def test_existing_algorithms(self):
        x = torch.empty((2, 96), dtype=torch.float16, device="meta")
        for algorithm, dst in ((0, 296), (0, 297), (0, 291), (0, 292), (1, 292), (2, 296)):
            with self.subTest(algorithm=algorithm, dst=dst):
                y, scale = torch_npu.npu_dynamic_mx_quant(x, scale_alg=algorithm, dst_type=dst)
                self.assertEqual(tuple(y.shape), (2, 48 if dst in (296, 297) else 96))
                self.assertEqual(tuple(scale.shape), (2, 2, 2))
                with self.assertRaisesRegex(RuntimeError, "divisible by 32"):
                    torch_npu.npu_dynamic_mx_quant(x, scale_alg=algorithm, dst_type=dst, block_size=16)


class TestDynamicMxQuantNvfp4(TestCase):
    @SupportedDevices(["Ascend950"])
    def test_nvfp4_fp32_rejected_by_backend(self):
        x = torch.zeros((2, 16), dtype=torch.float32, device="npu")
        with self.assertRaises(RuntimeError):
            torch_npu.npu_dynamic_mx_quant(x, scale_alg=3, block_size=16)

    def check_source(self, source, mode="rint", max_low_bound=0.0, strided=False):
        expected = nvfp4_reference(source, mode, max_low_bound)
        if strided:
            storage = torch.empty((*source.shape[:-1], source.shape[-1] * 2),
                                  dtype=source.dtype, device="npu")
            x = storage[..., 1::2]
            x.copy_(source)
        else:
            x = source.npu()
        actual = torch_npu.npu_dynamic_mx_quant(
            x, scale_alg=3, block_size=16, round_mode=mode, max_low_bound=max_low_bound)
        for result, reference in zip(actual, expected):
            self.assertEqual(result.dtype, torch.uint8)
            self.assertEqual(tuple(result.shape), tuple(reference.shape))
            self.assertTrue(torch.equal(result.cpu(), reference))

    @SupportedDevices(["Ascend950"])
    def test_nvfp4_precision(self):
        generator = torch.Generator().manual_seed(20261110)
        shapes = ((16,), (4, 18), (3, 34), (2, 2, 66), (2, 2, 2, 130),
                  (2, 2, 2, 2, 258), (2, 2, 2, 2, 2, 2, 16), (8, 4096))
        for dtype, mode, shape in itertools.product(
                (torch.float16, torch.bfloat16), ("rint", "round", "floor"), shapes):
            info = torch.finfo(dtype)
            smallest = torch.nextafter(torch.tensor(0, dtype=dtype), torch.tensor(1, dtype=dtype)).item()
            profiles = (
                torch.randn(shape, generator=generator),
                torch.zeros(shape), torch.full(shape, -0.0), torch.full(shape, float("nan")),
                torch.tensor([0., -0., float("nan"), float("inf"), -float("inf"),
                              info.max, -info.max, info.tiny, -info.tiny, smallest, -smallest]),
                torch.tensor([0.25, -0.25, 0.75, -0.75, 1.25, -1.25, 1.75, -1.75,
                              2.5, -2.5, 3.5, -3.5, 5., -5., 6., -6.]),
                (torch.rand(shape, generator=generator) * 2 - 1) * info.max,
            )
            for profile_index, values in enumerate(profiles):
                with self.subTest(dtype=dtype, mode=mode, shape=shape, profile=profile_index):
                    source = values.flatten().repeat(math.ceil(math.prod(shape) / values.numel()))
                    self.check_source(source[:math.prod(shape)].reshape(shape).to(dtype), mode)

    @SupportedDevices(["Ascend950"])
    def test_nvfp4_views_empty_and_lower_bound(self):
        for dtype, shape in itertools.product((torch.float16, torch.bfloat16),
                                               ((2, 18), (2, 2, 2, 34), (2, 2, 2, 2, 66))):
            with self.subTest(dtype=dtype, shape=shape):
                source = torch.linspace(-2, 2, math.prod(shape)).reshape(shape).to(dtype)
                self.check_source(source, max_low_bound=12.0, strided=True)
                self.check_source(torch.empty((*shape[:-1], 0), dtype=dtype))


if __name__ == "__main__":
    run_tests()
