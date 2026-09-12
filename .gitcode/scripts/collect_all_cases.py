#!/usr/bin/env python3
"""
Collect all test cases and split into shards.

This script runs in prepare job (once) to:
1. Discover test files by type (distributed/regular)
2. Collect all test cases via pytest --collect-only
3. Split cases evenly into N shards
4. Output shard JSON files for each type
5. Save collection error logs for failed files

Usage:
    python collect_all_cases.py \
        --test-dir /path/to/pytorch/test \
        --case-paths-config /path/to/case_paths_ci.yml \
        --distributed-shards 2 \
        --regular-npu-shards 5 \
        --regular-cpu-shards 3 \
        --output-dir /path/to/output \
        --error-log-dir /path/to/error_logs \
        --parallel 16
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

# Import discover_test_files module
import discover_test_files


def _normalize_test_file_path(test_file: str) -> str:
    """
    Remove 'test/' prefix from test file path if present.

    Args:
        test_file: Test file path (e.g., "test/distributed/pipelining/test_backward.py")

    Returns:
        Relative path without 'test/' prefix
    """
    if test_file.startswith("test/"):
        return test_file[5:]
    return test_file


def get_test_file_parent_dir(test_file: str, test_dir: Path) -> Path:
    """
    Get the parent directory of a test file.

    This directory should be added to PYTHONPATH to enable
    imports of sibling modules (e.g., model_registry.py).

    Args:
        test_file: Test file path (e.g., "test/distributed/pipelining/test_backward.py")
        test_dir: Path to PyTorch test directory

    Returns:
        Path to the test file's parent directory
    """
    test_file_rel = _normalize_test_file_path(test_file)
    test_file_path = Path(test_file_rel)
    return test_dir / test_file_path.parent


def collect_cases_for_file(test_file: str, test_dir: Path) -> Tuple[str, str, List[str], bool, str]:
    """
    Collect test cases from a single file.

    Adds test file's parent directory to PYTHONPATH to enable
    imports of sibling modules (e.g., 'from model_registry import MLPModule').

    Returns:
        Tuple of (test_file, display_name, nodeids, success, error_message)
        - test_file: Original test file path
        - display_name: Short name for logging (remove test/ prefix and .py suffix)
        - nodeids: List of collected test case nodeids
        - success: True if collection succeeded without errors
        - error_message: Error details if collection failed, empty string otherwise
    """
    test_file_rel = _normalize_test_file_path(test_file)

    # Extract display name (remove .py suffix)
    display_name = test_file_rel
    if display_name.endswith(".py"):
        display_name = display_name[:-3]

    # Get test file's parent directory for PYTHONPATH
    test_file_dir = get_test_file_parent_dir(test_file, test_dir)

    # Build environment with test file directory in PYTHONPATH
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(test_file_dir) + (":" + existing_pythonpath if existing_pythonpath else "")

    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "--quiet",
        test_file_rel,
    ]

    try:
        result = subprocess.run(
            command,
            cwd=str(test_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )

        nodeids = []
        for line in result.stdout.splitlines():
            stripped = line.strip()
            # pytest --collect-only -q outputs clean nodeids, one per line
            # Filter rules:
            # 1. Skip empty lines
            # 2. Skip summary lines (contain "collected" or "selected")
            # 3. Skip separator lines (start with "=")
            # 4. Must contain ".py::" to ensure it's a Python test file nodeid
            if not stripped:
                continue
            if "collected" in stripped or "selected" in stripped:
                continue
            if stripped.startswith("="):
                continue
            if ".py::" in stripped:
                nodeids.append(stripped)

        # Check for collection errors based on pytest exit codes:
        #   0: all passed (success)
        #   2: pytest error (includes collection errors like ImportError)
        #   3: all skipped (success)
        #   4: command line error (error)
        #   5: no tests collected (ERROR - test file should have cases)
        # Key insight: if a test file is selected for execution, it should have cases.
        # returncode 5 means 0 cases collected, which indicates a problem.
        if result.returncode in (0, 3):
            # Normal: passed or skipped
            return (test_file, display_name, nodeids, True, "")
        else:
            # returncode 2, 4, 5: real collection error
            # returncode 5 specifically means no tests collected - a problem for selected files
            error_msg = result.stdout.strip()
            if result.stderr.strip():
                error_msg += "\n--- stderr ---\n" + result.stderr.strip()

            # Diagnostic info for first failure: capture env state
            diag_lines = []
            try:
                diag_lines.append("--- Diagnostics ---")
                diag_lines.append("LD_LIBRARY_PATH: " + os.environ.get("LD_LIBRARY_PATH", "NOT SET"))
                diag_lines.append("PATH: " + os.environ.get("PATH", "NOT SET"))
                # 用原生文件遍历代替 find 子进程, 避免 bandit B607 告警
                libhccl_paths = list(Path("/usr/local/Ascend").rglob("libhccl.so")) if Path("/usr/local/Ascend").exists() else []
                diag_lines.append("find libhccl.so: " + (str(libhccl_paths[0]) if libhccl_paths else "NOT FOUND"))
                # 直接读文件代替 cat 子进程
                version_cfg = Path("/usr/local/Ascend/cann/version.cfg")
                diag_lines.append("CANN version: " + (version_cfg.read_text(encoding="utf-8").strip() if version_cfg.exists() else "MISSING"))
                # 当前进程直接 import torch 获取版本, 代替 python3 -c 子进程
                try:
                    import torch
                    diag_lines.append("torch version: torch: " + torch.__version__)
                except Exception as torch_err:
                    diag_lines.append("torch version: " + str(torch_err))
            except Exception:
                diag_lines.append("--- Diagnostics FAILED ---")
            error_msg += "\n" + "\n".join(diag_lines)

            return (test_file, display_name, nodeids, False, error_msg)

    except subprocess.TimeoutExpired:
        error_msg = f"TIMEOUT: Collection took >120s for {display_name}"
        return (test_file, display_name, [], False, error_msg)
    except Exception as e:
        error_msg = f"ERROR: {e}"
        return (test_file, display_name, [], False, error_msg)


def collect_all_cases(
    test_files: List[str],
    test_dir: Path,
    error_log_dir: Path,
    parallel: int = 16,
) -> List[Dict]:
    """
    Collect all cases from all files.

    Args:
        test_files: List of test file paths
        test_dir: Path to PyTorch test directory
        error_log_dir: Directory to save error logs for failed collections
        parallel: Number of parallel workers

    Returns:
        List of dicts with nodeid and file for each collected case
    """
    all_cases = []
    failed_files = []  # Track files with collection errors for logging

    print(f"Collecting cases from {len(test_files)} files with {parallel} workers...")
    print("=" * 60)

    # Create error log directory
    error_log_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(collect_cases_for_file, f, test_dir): f
            for f in test_files
        }

        completed = 0
        successful_count = 0
        failed_count = 0
        total_cases = 0

        for future in as_completed(futures):
            test_file, display_name, nodeids, success, error_msg = future.result()
            completed += 1

            if success:
                successful_count += 1
                # Print concise log for successful files
                print(f"  {display_name}: {len(nodeids)} cases")
                for nodeid in nodeids:
                    all_cases.append({
                        "nodeid": nodeid,
                        "file": test_file,
                    })
            else:
                failed_count += 1
                # Print concise log for failed files
                print(f"  [FAILED] {display_name}: {len(nodeids)} cases")
                # Save error details to log file
                failed_files.append({
                    "file": display_name,
                    "error": error_msg,
                    "cases": len(nodeids),
                    "test_file": test_file,
                })
                # Still add any cases that were collected despite errors
                for nodeid in nodeids:
                    all_cases.append({
                        "nodeid": nodeid,
                        "file": test_file,
                    })

            # Update total cases count for progress display
            total_cases += len(nodeids)

            # Print progress summary every 100 files
            if completed % 100 == 0:
                print(f"  [Progress: {completed}/{len(test_files)} files, {successful_count} ok, {failed_count} failed, {total_cases} cases]")

    print("=" * 60)

    # Save error logs to files
    if failed_files:
        save_error_logs(failed_files, error_log_dir)

    # Final summary
    print(f"Collection complete: {len(all_cases)} cases from {successful_count}/{len(test_files)} files")
    if failed_count > 0:
        print(f"  WARNING: {failed_count} files had collection errors (logs saved to {error_log_dir})")

    return all_cases


def save_error_logs(failed_files: List[Dict], error_log_dir: Path) -> None:
    """
    Save collection error logs to individual files and create a summary.

    Args:
        failed_files: List of dicts with file, error, cases info
        error_log_dir: Directory to save error logs
    """
    print(f"Saving error logs for {len(failed_files)} failed files...")

    # Save individual error log files
    for failed in failed_files:
        # Create safe filename from display name (replace / with _)
        safe_name = failed['file'].replace('/', '_')
        log_file = error_log_dir / f"{safe_name}.log"

        # Write error log
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"File: {failed['file']}\n")
            f.write(f"Cases collected: {failed['cases']}\n")
            f.write(f"Test file path: {failed['test_file']}\n")
            f.write("=" * 80 + "\n")
            f.write("Collection Error:\n")
            f.write("=" * 80 + "\n")
            f.write(failed['error'])
            f.write("\n")

    # Save summary JSON
    summary_file = error_log_dir / "collection_errors_summary.json"
    summary_data = {
        "total_failed": len(failed_files),
        "failed_files": [
            {
                "file": f['file'],
                "cases": f['cases'],
                "test_file": f['test_file'],
                "log_file": f"{f['file'].replace('/', '_')}.log",
            }
            for f in failed_files
        ],
    }
    summary_file.write_text(json.dumps(summary_data, indent=2), encoding='utf-8')

    print(f"  Error logs saved to {error_log_dir}")
    print(f"  Summary: {summary_file}")


def split_cases_into_shards(cases: List[Dict], num_shards: int) -> List[List[Dict]]:
    """Split cases evenly into shards."""
    total = len(cases)
    base_size = total // num_shards
    remainder = total % num_shards

    shards = []
    start = 0
    for i in range(num_shards):
        size = base_size + (1 if i < remainder else 0)
        shards.append(cases[start:start + size])
        start += size

    return shards


def save_cases_by_file(
    cases: List[Dict],
    test_files: List[str],
    test_type: str,
    output_dir: Path,
) -> Dict:
    """
    Save cases grouped by file in JSONL format.

    Includes all test files, even those with 0 cases collected.

    Output format (JSONL, one JSON object per line):
    Line 1: {"total_file":<count>,"total_cases":<count>}
    Line 2+: {"file_path":"...","case_count":<count>,"cases":["nodeid1","nodeid2",...]}
    """
    # Group cases by file
    file_groups: Dict[str, List[str]] = {}
    for case in cases:
        file_path = case["file"]
        if file_path not in file_groups:
            file_groups[file_path] = []
        file_groups[file_path].append(case["nodeid"])

    output_file = output_dir / f"{test_type}_cases_by_file.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        # Line 1: summary
        summary_line = json.dumps({
            "total_file": len(test_files),
            "total_cases": len(cases),
        }, separators=(',', ':'))
        f.write(summary_line + '\n')

        # Line 2+: file data (sorted by file path)
        for file_path in sorted(test_files):
            nodeids = file_groups.get(file_path, [])
            file_line = json.dumps({
                "file_path": file_path,
                "case_count": len(nodeids),
                "cases": nodeids,
            }, separators=(',', ':'))
            f.write(file_line + '\n')

    print(f"  Cases by file (JSONL): {len(test_files)} files -> {output_file}")

    return {
        "test_type": test_type,
        "total_files": len(test_files),
        "total_cases": len(cases),
    }


def save_shards(
    cases: List[Dict],
    num_shards: int,
    test_type: str,
    output_dir: Path,
) -> Dict:
    """Save shard JSONs and return summary."""
    shards = split_cases_into_shards(cases, num_shards)

    print(f"\nSaving {test_type} shards...")
    for i, shard_cases in enumerate(shards, 1):
        if num_shards == 1:
            shard_file = output_dir / f"{test_type}_cases.json"
        else:
            shard_file = output_dir / f"{test_type}_cases_shard_{i}.json"
        shard_data = {
            "shard": i,
            "num_shards": num_shards,
            "test_type": test_type,
            "total_cases": len(shard_cases),
            "cases": shard_cases,
        }
        shard_file.write_text(json.dumps(shard_data, indent=2), encoding="utf-8")
        print(f"  Shard {i}: {len(shard_cases)} cases -> {shard_file}")

    return {
        "test_type": test_type,
        "num_shards": num_shards,
        "total_cases": len(cases),
        "shard_sizes": [len(s) for s in shards],
    }


def _classify_regular_by_npu(cases: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Classify regular cases by NPU dependency based on class name.

    Cases whose class name ends with 'NPU' or 'PrivateUse1' require NPU hardware.
    All other cases can run on CPU-only machines.

    Returns:
        Dict with keys "npu" and "cpu", each mapping to a list of case dicts.
    """
    npu_cases = []
    cpu_cases = []
    for case in cases:
        nodeid = case.get("nodeid", "")
        parts = nodeid.split("::")
        # parts: [file_path, ClassName, method_name] or [file_path, method_name]
        class_name = parts[1] if len(parts) >= 3 else ""
        # Case-insensitive: PyTorch generates class suffixes like
        # 'TestModuleNPU' and 'TestModulePRIVATEUSE1' (all caps).
        class_name_upper = class_name.upper()
        if class_name_upper.endswith("NPU") or class_name_upper.endswith("PRIVATEUSE1"):
            npu_cases.append(case)
        else:
            cpu_cases.append(case)
    return {"npu": npu_cases, "cpu": cpu_cases}


def main():
    args = parse_args()

    test_dir = Path(args.test_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Error log directory for failed collections
    error_log_dir = Path(args.error_log_dir).resolve() if args.error_log_dir else output_dir / "collection_errors"
    error_log_dir.mkdir(parents=True, exist_ok=True)

    # ========================================
    # Step 1: Discover and classify test files (distributed / regular)
    # ========================================
    print("=" * 80)
    print("Discovering and classifying test files")
    print("=" * 80)

    dist_files, reg_files, discovery_meta = discover_test_files.discover_classified(
        test_dir=test_dir,
        case_paths_config=args.case_paths_config,
    )
    print(f"Total files scanned: {discovery_meta['total_files']}")
    print(f"Distributed: {discovery_meta['distributed_total']} files "
          f"(directory: {discovery_meta['dir_distributed']}, "
          f"style-detected: {discovery_meta['style_distributed']})")
    print(f"Regular: {discovery_meta['regular_total']} files")

    # ========================================
    # Step 2: Collect distributed test cases (serial execution later)
    # ========================================
    print("\n" + "=" * 80)
    print("Collecting distributed test cases")
    print("=" * 80)

    if dist_files:
        all_dist_cases = collect_all_cases(dist_files, test_dir, error_log_dir / "distributed", args.parallel)
        print(f"Total distributed cases: {len(all_dist_cases)}")
    else:
        all_dist_cases = []
        print("No distributed test files found, skipping distributed collection")

    # ========================================
    # Step 3: Collect regular test cases (parallel execution later)
    # ========================================
    print("\n" + "=" * 80)
    print("Collecting regular test cases")
    print("=" * 80)

    if reg_files:
        reg_cases = collect_all_cases(reg_files, test_dir, error_log_dir / "regular", args.parallel)
        print(f"Total regular cases: {len(reg_cases)}")
    else:
        reg_cases = []
        print("No regular test files found, skipping regular collection")

    # ========================================
    # Step 4: Shard distributed cases (fallback serial mode, no card split)
    # ========================================
    if all_dist_cases:
        dist_summary = save_shards(all_dist_cases, args.distributed_shards, "distributed", output_dir)
        save_cases_by_file(all_dist_cases, dist_files, "distributed", output_dir)
    else:
        dist_summary = {
            "test_type": "distributed",
            "num_shards": 0,
            "total_cases": 0,
            "shard_sizes": [],
        }

    # ========================================
    # Step 5: Shard regular cases (parallel mode)
    # ========================================
    if reg_cases:
        if args.regular_cpu_shards > 0:
            # 显式开启 cpu 划分时才按类名后缀区分 npu/cpu
            npu_groups = _classify_regular_by_npu(reg_cases)
            npu_shard_map = {"npu": args.regular_npu_shards, "cpu": args.regular_cpu_shards}
            reg_summaries = {}
            for label, cases in sorted(npu_groups.items()):
                print(f"  [{label}] {len(cases)} cases (class ends with NPU/PrivateUse1)" if label == "npu"
                      else f"  [{label}] {len(cases)} cases (no NPU required)")
                if cases:
                    test_type = f"regular_{label}"
                    num_shards = npu_shard_map.get(label, args.regular_npu_shards)
                    s = save_shards(cases, num_shards, test_type, output_dir)
                    save_cases_by_file(cases, reg_files, test_type, output_dir)
                    reg_summaries[label] = s
                    print(f"  [{label}] -> {num_shards} shards")
        else:
            # 默认: 不区分 cpu, 所有常规用例均视为 NPU 用例,
            # 统一走 regular_npu 分片 (并行执行)
            print(f"  [npu] {len(reg_cases)} cases "
                  f"(all regular cases treated as NPU, no cpu/npu split)")
            test_type = "regular_npu"
            num_shards = args.regular_npu_shards
            s = save_shards(reg_cases, num_shards, test_type, output_dir)
            save_cases_by_file(reg_cases, reg_files, test_type, output_dir)
            reg_summaries = {"npu": s}
            print(f"  [npu] -> {num_shards} shards")

        reg_summary = {
            "test_type": "regular",
            "num_shards": sum(s["num_shards"] for s in reg_summaries.values()),
            "total_cases": len(reg_cases),
            "groups": {
                label: {"num_shards": s["num_shards"], "total_cases": s["total_cases"],
                        "shard_sizes": s["shard_sizes"]}
                for label, s in reg_summaries.items()
            },
        }
    else:
        reg_summary = {
            "test_type": "regular",
            "num_shards": 0,
            "total_cases": 0,
            "shard_sizes": [],
        }

    # ========================================
    # Step 6: Save overall summary
    # ========================================
    total_files = discovery_meta.get("total_files", 0)

    overall_summary = {
        "distributed": {
            "cases_summary": dist_summary,
            "discovery_metadata": discovery_meta,
            "style_detected_files": discovery_meta.get("style_distributed_files", []),
        },
        "regular": {
            "cases_summary": reg_summary,
        },
        "total_cases": len(all_dist_cases) + len(reg_cases),
        "total_files_scanned": total_files,
        "distributed_files": discovery_meta.get("distributed_total", 0),
        "regular_files": discovery_meta.get("regular_total", 0),
    }
    summary_file = output_dir / "cases_collection_summary.json"
    summary_file.write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")
    print(f"\nOverall summary saved to {summary_file}")

    print("\n" + "=" * 80)
    print("Collection Complete")
    print("=" * 80)
    print(f"Distributed: {len(all_dist_cases)} cases -> {dist_summary.get('num_shards', 0)} shards "
          f"(serial execution)")
    print(f"Regular: {len(reg_cases)} cases -> {reg_summary.get('num_shards', 0)} shards (parallel execution)")
    if "groups" in reg_summary:
        for label, rg in sorted(reg_summary["groups"].items()):
            print(f"  [{label}] {rg['total_cases']} cases -> {rg['num_shards']} shards")
    print(f"Total: {len(all_dist_cases) + len(reg_cases)} cases")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect and shard test cases")
    parser.add_argument("--test-dir", required=True, help="PyTorch test directory")
    parser.add_argument("--case-paths-config", help="case_paths_ci.yml path")
    parser.add_argument("--distributed-shards", type=int, default=5, help="Distributed test shards")
    parser.add_argument("--regular-npu-shards", type=int, default=5, help="Regular NPU test shards")
    parser.add_argument("--regular-cpu-shards", type=int, default=0, help="Regular CPU test shards (0 = no cpu/npu split, all regular cases treated as NPU)")
    parser.add_argument("--output-dir", required=True, help="Output directory for shard JSONs")
    parser.add_argument("--error-log-dir", help="Output directory for collection error logs (default: output-dir/collection_errors)")
    parser.add_argument("--parallel", type=int, default=16, help="Parallel collection workers")
    return parser.parse_args()


if __name__ == "__main__":
    main()
