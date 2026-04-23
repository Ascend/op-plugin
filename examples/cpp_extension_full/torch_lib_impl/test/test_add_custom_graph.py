from typing import Any
import torch
from torch.library import impl, Library
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import cpp_extension_full


class TestTorchCompileCustomAdd(TestCase):

    def test_add_custom(self):
        x = torch.randn([8, 2048], device='npu', dtype=torch.float16)
        y = torch.randn([8, 2048], device='npu', dtype=torch.float16)

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                result = cpp_extension_full.ops.add_custom(x, y)
                return result
        mod = torch.compile(Module().npu(), backend="npugraph_ex")

        output = mod(x, y)
        self.assertRtolEqual(output, (x + y))


if __name__ == "__main__":
    run_tests()
