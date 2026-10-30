"""
AxiomLM Benchmark Hardware & Environment Telemetry Collector.
Captures host hardware, SoC characteristics, operating system, and framework versions
to guarantee full experimental reproducibility for academic publication.
"""

import os
import sys
import json
import subprocess
import platform
import shutil
from typing import Dict, Any


def get_command_output(cmd: list) -> str:
    """Executes a command and returns stripped output, or empty string on failure."""
    try:
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        return res.stdout.strip()
    except Exception:
        return ""


def get_git_info() -> Dict[str, str]:
    """Retrieves current Git commit hash, branch, and working tree dirty status."""
    commit = get_command_output(["git", "rev-parse", "HEAD"])
    branch = get_command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = get_command_output(["git", "status", "--porcelain"])
    return {
        "commit_hash": commit or "unknown",
        "branch": branch or "unknown",
        "is_dirty": bool(status),
    }


def get_mac_hardware_telemetry() -> Dict[str, Any]:
    """Captures Apple Silicon SoC architecture, CPU/GPU core counts, and unified memory."""
    telemetry: Dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": (
            platform.mac_ver()[0] if hasattr(platform, "mac_ver") else "unknown"
        ),
        "arch": platform.machine(),
        "soc_name": "Apple Silicon",
        "cpu_cores": os.cpu_count() or 0,
        "gpu_name": "Apple Metal GPU",
        "gpu_cores": 0,
        "ram_bytes": 0,
        "ram_gb": 0.0,
        "peak_bandwidth_gb_s": 150.0,  # Standard baseline for M3 Pro
    }

    # Brand string
    cpu_brand = get_command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    if cpu_brand:
        telemetry["soc_name"] = cpu_brand

    # Physical memory
    mem_str = get_command_output(["sysctl", "-n", "hw.memsize"])
    if mem_str.isdigit():
        bytes_val = int(mem_str)
        telemetry["ram_bytes"] = bytes_val
        telemetry["ram_gb"] = round(bytes_val / (1024**3), 2)

    # GPU profile via system_profiler
    display_info = get_command_output(["system_profiler", "SPDisplaysDataType"])
    for line in display_info.splitlines():
        line_s = line.strip()
        if "Chipset Model:" in line_s:
            telemetry["gpu_name"] = line_s.split("Chipset Model:")[-1].strip()
        elif "Total Number of Cores:" in line_s:
            cores = line_s.split("Total Number of Cores:")[-1].strip()
            if cores.isdigit():
                telemetry["gpu_cores"] = int(cores)

    # Estimate theoretical peak memory bandwidth based on Apple Silicon SoC
    soc = telemetry["soc_name"].lower()
    if "m3 max" in soc or "m4 max" in soc:
        telemetry["peak_bandwidth_gb_s"] = 400.0
    elif "m3 pro" in soc or "m4 pro" in soc:
        telemetry["peak_bandwidth_gb_s"] = 150.0
    elif "m2 max" in soc or "m1 max" in soc:
        telemetry["peak_bandwidth_gb_s"] = 400.0
    elif "m2 pro" in soc or "m1 pro" in soc:
        telemetry["peak_bandwidth_gb_s"] = 200.0
    elif "ultra" in soc:
        telemetry["peak_bandwidth_gb_s"] = 800.0
    elif "m3" in soc or "m2" in soc or "m1" in soc or "m4" in soc:
        telemetry["peak_bandwidth_gb_s"] = 100.0

    return telemetry


def get_software_stack_versions() -> Dict[str, str]:
    """Captures versions of all installed ML compilers and runtime libraries."""
    versions: Dict[str, str] = {
        "python": sys.version.split()[0],
    }

    # PyTorch
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["mps_available"] = str(torch.backends.mps.is_available())
        versions["mps_built"] = str(torch.backends.mps.is_built())
    except ImportError:
        versions["torch"] = "not_installed"

    # MLX
    try:
        import mlx.core as mx

        versions["mlx"] = getattr(mx, "__version__", "installed")
        versions["mlx_device"] = str(mx.default_device())
    except ImportError:
        versions["mlx"] = "not_installed"

    # llama-bench / llama.cpp
    llama_bench_path = shutil.which("llama-bench")
    if llama_bench_path:
        versions["llama_bench_path"] = llama_bench_path
        ver_output = get_command_output([llama_bench_path, "--version"])
        versions["llama_bench_version"] = (
            ver_output.splitlines()[0] if ver_output else "installed"
        )
    else:
        versions["llama_bench_path"] = "not_found"

    return versions


def collect_system_telemetry() -> Dict[str, Any]:
    """Collects complete reproducibility manifest."""
    return {
        "hardware": get_mac_hardware_telemetry(),
        "software": get_software_stack_versions(),
        "git": get_git_info(),
    }


if __name__ == "__main__":
    data = collect_system_telemetry()
    print(json.dumps(data, indent=2))
