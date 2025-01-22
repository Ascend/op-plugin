# -*- coding: UTF-8 -*-

import os
import re
import sys
import subprocess
import threading
import queue
from abc import ABCMeta, abstractmethod
from pathlib import Path

import torch_npu


BASE_DIR = Path(__file__).absolute().parent.parent
TEST_DIR = os.path.join(BASE_DIR, 'test')

modify_file_hash = {}

not_support_in_910b = [
    "test_custom_ops/test_incre_flash_attention",
    "test_custom_ops/test_npu_ffn",
    "test_base_ops/test_adaptive_max_pool2d_backward",
    "test_base_ops/test_random",
    "test_base_ops/test_im2col_backward",
    "test_base_ops/test_conv_transpose2d_backward",
    "test_base_ops/test_gru_true",
]


class AccurateTest(metaclass=ABCMeta):
    @abstractmethod
    def identify(self, modify_file):
        """
        This interface provides the path information for the corresponding unit tests in the code.
        """
        raise Exception("abstract method. Subclasses should implement it.")

    @staticmethod
    def find_ut_by_regex(regex, test_path):
        ut_files = []
        cmd = "find {} -name {}".format(test_path, regex)
        status, output = subprocess.getstatusoutput(cmd)
        # For the ones that cannot be found, no action will be taken temporarily.
        if status or not output:
            pass
        else:
            files = output.split('\n')
            for ut_file in files:
                if ut_file.endswith(".py"):
                    ut_files.append(ut_file)
        return ut_files

    def get_ut_files(self, regex):
        test_custom_path = os.path.join(TEST_DIR, 'test_custom_ops')
        check_dir_path_readable(test_custom_path)
        ut_files = self.find_ut_by_regex(regex, test_custom_path)
        
        version_path = get_test_torch_version_path()
        test_version_path = os.path.join(TEST_DIR, version_path)
        check_dir_path_readable(test_version_path)
        version_ut_files = self.find_ut_by_regex(regex, test_version_path)
        if not version_ut_files:
            test_base_path = os.path.join(TEST_DIR, 'test_base_ops')
            check_dir_path_readable(test_base_path)
            base_ut_files = self.find_ut_by_regex(regex, test_base_path)
            ut_files.extend(base_ut_files)
        else:
            for file in version_ut_files:
                file_name = Path(file).name
                modify_file_hash[file_name] = file
        return ut_files


class OpStrategy(AccurateTest):
    """
    Identifying the code of the adaptation layer.
    """
    def identify(self, modify_file):
        """
        By parsing the file name of the operator implementation file to obtain its unit test name, for example:
        BinaryCrossEntropyWithLogitsBackwardKernelNpu.cpp
        For this file, first identify the keywords: BinaryCrossEntropyWithLogitsBackward
        Then, use the regular expression *binary*cross*entropy*with*logits*backward* to identify
            the matching test case.
        Specific method: Split the keywords using capital letters and then identify
            the test file names that contain all of these keywords.
        """
        filename = Path(modify_file).name
        if filename.find('KernelNpu') >= 0: 
            feature_line = filename.split('KernelNpu')[0]
            features = re.findall('[A-Z][^A-Z]*', feature_line)
            regex = '*_' + '*'.join([f"{feature.lower()}" for feature in features]) + '.py'
            return self.get_ut_files(regex)
        return []


class DirectoryStrategy(AccurateTest):
    """
    Determine whether the modified files are test cases
    """
    def identify(self, modify_file, ut_files):
        is_test_file = str(Path(modify_file).parts[0]) == "test" \
            and re.match("test_(.+).py", Path(modify_file).name)
        version_path = get_test_torch_version_path()
        if is_test_file and str(Path(modify_file).parts[1]) in [version_path, "test_custom_ops", "test_base_ops"]:
            modify_file_path = os.path.join(BASE_DIR, modify_file)
            modify_file_name = Path(modify_file).name
            if modify_file_name in modify_file_hash:
                if str(Path(modify_file).parts[1]) == version_path:
                    modify_file_hash[modify_file_name] = modify_file_path
            else:
                modify_file_hash[modify_file_name] = modify_file_path


class CoreTestStrategy(AccurateTest):
    """
    Determine whether the core tests should be runned
    """
    block_list = ['test', 'docs']
    core_test_cases = [str(i) for i in (BASE_DIR / 'test/core_tests').rglob('test_*.py')]

    def identify(self, modify_file):
        modified_module = str(Path(modify_file).parts[0])
        if modified_module not in self.block_list:
            return self.core_test_cases
        return []


class TestMgr():
    def __init__(self):
        self.modify_files = []
        self.test_files = {
            'ut_files': [],
            'op_ut_files': []
        }

    def load(self, modify_files):
        check_dir_path_readable(modify_files)
        with open(modify_files) as f:
            for line in f:
                line = line.strip()
                self.modify_files.append(line)

    def analyze(self):
        for modify_file in self.modify_files:
            DirectoryStrategy().identify(modify_file, self.test_files['ut_files'])
            self.test_files['ut_files'] += OpStrategy().identify(modify_file)
            self.test_files['ut_files'] += CoreTestStrategy().identify(modify_file)
        for v in modify_file_hash.values():
            self.test_files['ut_files'].append(v)
        unique_files = sorted(set(self.test_files['ut_files']))

        exist_ut_file = [
            changed_file
            for changed_file in unique_files
            if Path(changed_file).exists()
        ]
        
        if "Ascend910B" in torch_npu.npu.get_device_name():
            supported_ut_files = []
            for ut_file in exist_ut_file:
                if ut_file.split('test/')[-1][:-3] in not_support_in_910b:
                    print(ut_file, "can not run in Ascend910B, skip it now.")
                else:
                    supported_ut_files.append(ut_file)
            self.test_files['ut_files'] = supported_ut_files
        else:
            self.test_files['ut_files'] = exist_ut_file

    def get_test_files(self):
        return self.test_files

    def print_modify_files(self):
        print("modify files:")
        for modify_file in self.modify_files:
            print(modify_file)

    def print_ut_files(self):
        print("ut files:")
        for ut_file in self.test_files['ut_files']:
            print(ut_file)
    
    def print_op_ut_files(self):
        print("op ut files:")
        for op_ut_file in self.test_files['op_ut_files']:
            print(op_ut_file)


def exec_ut(files):
    """
    Execute the unit test file, and if there are any failures, identify
        the exceptions and print relevant information.
    """
    def get_op_name(ut_file):
        return ut_file.split('/')[-1].split('.')[0].lstrip('test_')
    
    def get_ut_name(ut_file):
        return str(Path(ut_file).relative_to(TEST_DIR))[:-3]

    def get_ut_cmd(ut_type, ut_file):
        cmd = [sys.executable, "run_test.py", "-v", "-i"]
        if ut_type == "op_ut_files":
            return cmd + ["test_ops", "--", "-k", get_op_name(ut_file)]
        return cmd + [get_ut_name(ut_file)]

    def wait_thread(process, event_timer):
        process.wait()
        event_timer.set()

    def enqueue_output(out, log_queue):
        for line in iter(out.readline, b''):
            log_queue.put(line.decode('utf-8'))
        out.close()
        return

    def start_thread(fn, *args):
        stdout_t = threading.Thread(target=fn, args=args)
        stdout_t.daemon = True
        stdout_t.start()

    def print_subprocess_log(log_queue):
        while (not log_queue.empty()):
            print((log_queue.get()).strip())

    def run_cmd_with_timeout(cmd):
        os.chdir(str(TEST_DIR))
        stdout_queue = queue.Queue()
        event_timer = threading.Event()

        p = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=False)
        start_thread(wait_thread, p, event_timer)
        start_thread(enqueue_output, p.stdout, stdout_queue)
        event_timer.wait(2000)
        ret = p.poll()
        if ret:
            print_subprocess_log(stdout_queue)
        if not event_timer.is_set():
            ret = 1
            p.kill()
            p.terminate()
            print("Timeout: Command '{}' timed out after 2000 seconds".format(" ".join(cmd)))
            print_subprocess_log(stdout_queue)

        return ret

    def run_tests(files):
        exec_infos = []
        has_failed = 0
        for ut_type, ut_files in files.items():
            for ut_file in ut_files:
                cmd = get_ut_cmd(ut_type, ut_file)
                ut_info = " ".join(cmd[4:]).replace(" -- -k", "")
                ret = run_cmd_with_timeout(cmd)
                if ret:
                    has_failed = ret
                    exec_infos.append("exec ut {} failed.".format(ut_info))
                else:
                    exec_infos.append("exec ut {} success.".format(ut_info))
        return has_failed, exec_infos

    ret_status, exec_infos = run_tests(files)

    print("***** Total result:")
    for exec_info in exec_infos:
        print(exec_info)
    return ret_status


def check_dir_path_readable(file_path):
    """
    check file path readable.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The path does not exist: {file_path}")
    if os.stat(file_path).st_uid != os.getuid():
        check_msg = input("The path does not belong to you, do you want to continue? [y/n]")
        if check_msg.lower() != 'y':
            raise RuntimeError("The user choose not to contiue")
    if os.path.islink(file_path):
        raise RuntimeError(f"Invalid path is a soft chain: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"The path permission check filed: {file_path}")


def get_test_torch_version_path():
    torch_npu_version = torch_npu.__version__
    version_list = torch_npu_version.split('.')
    if len(version_list) > 2:
        return f'test_v{version_list[0]}r{version_list[1]}_ops'
    else:
        raise RuntimeError("Invalid torch_npu version.")


if __name__ == "__main__":
    cur_modify_files = os.path.join(BASE_DIR, 'modify_files.txt')
    test_mgr = TestMgr()
    test_mgr.load(cur_modify_files)
    test_mgr.analyze()
    cur_test_files = test_mgr.get_test_files()

    test_mgr.print_modify_files()
    test_mgr.print_ut_files()
    test_mgr.print_op_ut_files()

    ret_ut = exec_ut(cur_test_files)
    sys.exit(ret_ut)
