from typing import Any
import torch
from torch.library import impl, Library
import torch_npu
import torchair
from torch_npu.testing.testcase import TestCase, run_tests
from torchair import register_fx_node_ge_converter
from torchair.ge import Tensor
import custom_ops


# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.myops.add_custom.default)
def convert_npu_add_custom(x: Tensor, y: Tensor, z: Tensor = None, meta_outputs: Any = None):
    return torchair.ge.custom_op(
        "AddCustom",
        inputs={"x": x, "y": y},
        outputs=['z']
    )


class TestTorchCompileCustomAdd(TestCase):

    def test_add_custom(self):
        x = torch.randn([8, 2048], device='npu', dtype=torch.float16)
        y = torch.randn([8, 2048], device='npu', dtype=torch.float16)

        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                result = custom_ops.add_custom(x, y)
                return result
        mod = torch.compile(Module().npu(), backend=npu_backend)

        output = mod(x, y)
        self.assertRtolEqual(output, (x + y))


if __name__ == "__main__":
    run_tests()
