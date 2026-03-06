import os
import unittest
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuSaveTensor(TestCase):
    @unittest.skip("skip until CANN is updated to support aclrtLaunchHostFunc")
    def test_save_npugraph_tensor_bin(self):
        x = torch.randn([5, 5]).npu()
        torch_npu.save_npugraph_tensor(x, save_path="./x.bin")
        
        self.assertTrue(os.path.exists("./x_device_0.bin"))
        actual = torch.load("./x_device_0.bin")
        self.assertEqual(x.dtype, actual.dtype)
        self.assertEqual(x.shape, actual.shape)
        self.assertRtolEqual(x, actual, 0.001)
    
    @unittest.skip("skip until CANN is updated to support aclrtLaunchHostFunc")
    def test_save_npugraph_tensor_pt(self):
        x = torch.randn([10, 10]).npu()
        torch_npu.save_npugraph_tensor(x, save_path="./x.pt")
        
        self.assertTrue(os.path.exists("./x_device_0.pt"))
        actual = torch.load("./x_device_0.pt")
        self.assertEqual(x.dtype, actual.dtype)
        self.assertEqual(x.shape, actual.shape)
        self.assertRtolEqual(x, actual, 0.001)
    
    @unittest.skip("skip until CANN is updated to support aclrtLaunchHostFunc")
    def test_save_npugraph_tensor_empty(self):
        x = torch.randn([6, 0]).npu()
        torch_npu.save_npugraph_tensor(x, save_path="./x.pt")
        
        self.assertTrue(os.path.exists("./x_device_0.pt"))
        actual = torch.load("./x_device_0.pt")
        self.assertEqual(x.dtype, actual.dtype)
        self.assertEqual(x.shape, actual.shape)
    
    @unittest.skip("skip until CANN is updated to support aclrtLaunchHostFunc")
    def test_save_npugraph_tensor_replay(self):
        x = torch.randn([6, 0]).npu()
        graph1 = torch.npu.NPUGraph()
        with torch.npu.graph(graph1):
            output = torch.square(x)
            torch.ops.npu.save_npugraph_tensor(output, save_path="./out.bin")
        
        for _ in range(3):
            graph1.replay()
        
        self.assertTrue(os.path.exists("./out_device_0.bin"))
    
    @unittest.skip("skip until CANN is updated to support aclrtLaunchHostFunc")
    def test_save_npugraph_tensor_reduce_overhead(self):
        import torchair
        from torchair.configs.compiler_config import CompilerConfig
        
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x0, x1):
                torch_npu.save_npugraph_tensor(x0, save_path="./x0.pt")
                torch_npu.save_npugraph_tensor(x1, save_path="./x1.pt")
                sq1 = torch.square(x0)
                torch_npu.save_npugraph_tensor(sq1, save_path="./sq1.pt")
                add1 = torch.add(x1, sq1)
                torch_npu.save_npugraph_tensor(add1, save_path="./add1.pt")
                return add1
            
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        
        x0 = torch.randn([6, 6]).npu()
        x1 = torch.randn([6, 6]).npu()
        
        model = Model()
        model = torch.compile(model, backend=npu_backend)
        model(x0, x1)
        
        self.assertTrue(os.path.exists("./x0_device_0.bin"))
        self.assertTrue(os.path.exists("./x1_device_0.bin"))
        self.assertTrue(os.path.exists("./sq1_device_0.bin"))
        self.assertTrue(os.path.exists("./add1_device_0.bin"))
    
    @unittest.skip("skip until CANN is updated to support aclrtLaunchHostFunc")
    def test_save_npugraph_tensor_list(self):
        import torchair
        from torchair.configs.compiler_config import CompilerConfig
        
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x0, x1):
                tensor_tuple = [x0, x1]
                torch_npu.save_npugraph_tensor(tensor_tuple, "inputs")
                sq1 = torch.square(x0)
                add1 = torch.add(x1, sq1)
                torch_npu.save_npugraph_tensor([sq1, add1], "outputs")
                return add1
            
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        
        x0 = torch.randn([6, 6]).npu()
        x1 = torch.randn([6, 6]).npu()
        
        model = Model()
        model = torch.compile(model, backend=npu_backend)
        model(x0, x1)
        
        self.assertTrue(os.path.exists("./inputs_0_device_0.bin"))
        self.assertTrue(os.path.exists("./inputs_1_device_0.bin"))
        self.assertTrue(os.path.exists("./outputs_1_device_0.bin"))
        self.assertTrue(os.path.exists("./outputs_1_device_0.bin"))
    
    @unittest.skip("skip until CANN is updated to support aclrtLaunchHostFunc")
    def test_save_npugraph_tensor_tuple(self):
        import torchair
        from torchair.configs.compiler_config import CompilerConfig
        
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, query, key, value, head_dim, softmax_lse_flag):
                torch_npu.save_npugraph_tensor((query, key, value), "fia_inputs")
                scale = 1 / 0.0078125
                res = torch_npu.npu_fused_infer_attention_score(
                    query, key, value, num_heads=32, input_layout="BNSD", scale=scale, 
                    pre_tokens=65535, next_tokens=65535, softmax_lse_flag=softmax_lse_flag)
                torch_npu.save_npugraph_tensor(res, "fia_outputs")
                return res
            
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        
        query = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        key = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()
        value = torch.randn(1, 32, 2048, 128, dtype=torch.float16).npu()

        head_dim = 128
                
        model = Model()
        model = torch.compile(model, backend=npu_backend)
        model(query, key, value, head_dim, False)
        
        self.assertTrue(os.path.exists("./fia_inputs_0_device_0.bin"))
        self.assertTrue(os.path.exists("./fia_inputs_1_device_0.bin"))
        self.assertTrue(os.path.exists("./fia_inputs_2_device_0.bin"))
        self.assertTrue(os.path.exists("./fia_outputs_0_device_0.bin"))
        self.assertTrue(os.path.exists("./fia_outputs_1_device_0.bin"))
        self.assertTrue(os.path.exists("./fia_outputs_2_device_0.bin"))
    
    @unittest.skip("skip until CANN is updated to support aclrtLaunchHostFunc")
    def test_save_npugraph_tensor_multiple_calls(self):
        x = torch.randn([6, 0]).npu()
        graph1 = torch.npu.NPUGraph()
        with torch.npu.graph(graph1):
            output = torch.square(x)
            torch.ops.npu.save_npugraph_tensor(output, save_path="./out.bin")
            torch.ops.npu.save_npugraph_tensor(output, save_path="./out.bin")
            torch.ops.npu.save_npugraph_tensor(output, save_path="./out.bin")
        
        graph1.replay()
        
        self.assertTrue(os.path.exists("./out_device_0_0.bin"))
        self.assertTrue(os.path.exists("./out_device_0_1.bin"))
        self.assertTrue(os.path.exists("./out_device_0_2.bin"))


if __name__ == "__main__":
    run_tests()