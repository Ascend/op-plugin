import importlib
import inspect
import json
import os
import re
import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._utils_internal import get_file_path_2
import torch_npu


def set_failure_list(api_str, value, signature, failure_list):
    failure_list.append(f"# {api_str}:")
    failure_list.append(f"  - function signature is different: ")
    failure_list.append(f"    - the base signature is {value}.")
    failure_list.append(f"    - now it is {signature}.")


def is_not_compatibility(base_str, new_str):
    base_io_params = base_str.split("->")
    new_io_params = new_str.split("->")
    base_input_params = base_io_params[0].strip()
    new_input_params = new_io_params[0].strip()
    base_out_params = "" if len(base_io_params) == 1 else base_io_params[1].strip()
    new_out_params = "" if len(new_io_params) == 1 else new_io_params[1].strip()

    # output params
    if base_out_params != new_out_params:
        return True

    base_params = base_input_params[1:-1].split(",")
    new_params = new_input_params[1:-1].split(",")
    base_diff_params = set(base_params) - set(new_params)

    # case: delete/different default value/different parameter name/different parameter dtype
    if base_diff_params:
        return True

    new_diff_params = set(new_params) - set(base_params)
    for elem in new_diff_params:
        # case: add params
        if "=" not in elem:
            return True

    # case: position parameters
    base_arr = [elem for elem in base_params if "=" not in elem]
    new_arr = [elem for elem in new_params if "=" not in elem]
    i = 0
    while i < len(base_arr):
        if base_arr[i] != new_arr[i]:
            return True
        i += 1

    return False


def get_func_from_yaml(yaml_path):
    content = []
    with open(yaml_path, 'r') as f:
        for line in f.readlines():
            if " func:" in line:
                string = line.split("func:")[1].strip()
                content.append(string)
    return content


def parse_func_str(func_str):
    if "(" in func_str:
        func_name = func_str.split("(")[0]
        signature = func_str.split(func_name)[1]
    else:
        func_name = func_str
        signature = ""
    return func_name, signature


def func_from_yaml(content, base_schema, failure_list):
    torch_npu_path = torch_npu.__path__[0]
    yaml1_path = os.path.join(torch_npu_path, "csrc/aten/npu_native_functions_by_codegen.yaml")
    yaml2_path = os.path.join(torch_npu_path, "csrc/aten/npu_native_functions.yaml")
    yaml1_content = get_func_from_yaml(yaml1_path)
    yaml2_content = get_func_from_yaml(yaml2_path)

    op_funcs = set(yaml1_content) - set(yaml2_content)

    for func in op_funcs:
        func_name, signature = parse_func_str(func)
        if "func: " + func_name in base_schema:
            value = base_schema["func: " + func_name]["signature"]
            if is_not_compatibility(value, signature):
                set_failure_list("func: " + func_name, value, signature, failure_list)
        content["func: " + func_name] = {"signature": signature}


def _get_test_torch_version():
    torch_npu_version = torch_npu.__version__
    version_list = torch_npu_version.split('.')
    if len(version_list) > 2:
        return f'v{version_list[0]}.{version_list[1]}'
    else:
        raise RuntimeError("Invalid torch_npu version.")


class TestOpApiCompatibility(TestCase):

    @unittest.skipIf(_get_test_torch_version() in ["v1.11", "v2.0", "v2.2"],
                     "Skipping test for these torch versions.")
    def test_op_api_compatibility(self):
        failure_list = []

        # load torch_npu_OpApi_schema_all.json
        with open(get_file_path_2(os.path.dirname(__file__), "torch_npu_OpApi_schema_all.json")) as fp:
            all_base_schema = json.load(fp)
        version_tag = _get_test_torch_version()
        base_schema = {}
        for key, value in all_base_schema.items():
            if not("all_version" in value['version'] or version_tag in value['version']):
                continue
            func_name, signature = parse_func_str(key)
            if func_name not in base_schema:
                base_schema[func_name] = dict()
                base_schema[func_name]["signature"] = signature

        content = {}

        # functions from npu_native_functions_by_codegen.yaml
        func_from_yaml(content, base_schema, failure_list)

        base_funcs = base_schema.keys()
        now_funcs = content.keys()
        deleted_apis = set(base_funcs) - set(now_funcs)
        for func in deleted_apis:
            failure_list.append(f"# {func}:")
            failure_list.append(f"  - {func} has been deleted.")

        msg = "All the OpAPIs below do not meet the compatibility guidelines. "
        msg += "If the change timeline has been reached, you can modify the torch_npu_OpApi_schema_all.json to make it OK."
        msg += "\n\nFull list:\n"
        msg += "\n".join(map(str, failure_list))
        # empty lists are considered false in python
        self.assertTrue(not failure_list, msg)


if __name__ == '__main__':
    run_tests()
