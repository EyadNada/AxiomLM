"""
Axiom-LM Kernel Compiler & Dynamic Loader.
Compiles and loads native C++/ARM NEON kernels optimized for Apple Silicon (M1/M2/M3 Pro/Max)
and configures fallback pathways.
"""

import os
import sys
import platform
import subprocess
import sysconfig
import importlib.util
from typing import Optional, Any
import torch

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_SOURCE = os.path.join(KERNEL_DIR, "cpu_neon_kernels.cpp")
SO_OUTPUT = os.path.join(KERNEL_DIR, "axiom_neon_kernels.so")

IS_ARM = platform.machine().lower() in ["arm64", "aarch64"]


def compile_neon_extension(force: bool = False, verbose: bool = False) -> str:
    """
    Compiles cpu_neon_kernels.cpp into a native shared library (.so)
    using clang++ with architecture-specific ARM SIMD optimizations.
    """
    if not IS_ARM:
        raise RuntimeError(
            f"ARM NEON kernels require an ARM64/aarch64 architecture (detected: {platform.machine()})"
        )

    if os.path.exists(SO_OUTPUT) and not force:
        # Check timestamp
        src_mtime = os.path.getmtime(CPP_SOURCE)
        so_mtime = os.path.getmtime(SO_OUTPUT)
        if so_mtime >= src_mtime:
            return SO_OUTPUT

    py_inc = sysconfig.get_path("include")
    torch_inc = os.path.join(os.path.dirname(torch.__file__), "include")
    torch_api_inc = os.path.join(torch_inc, "torch", "csrc", "api", "include")
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

    cmd = [
        "clang++",
        "-O3",
        "-Wall",
        "-shared",
        "-std=c++20",
        "-fPIC",
        "-ffast-math",
        "-DTORCH_EXTENSION_NAME=axiom_neon_kernels",
        f"-I{py_inc}",
        f"-I{torch_inc}",
        f"-I{torch_api_inc}",
        f"-L{torch_lib}",
        "-ltorch",
        "-ltorch_cpu",
        "-lc10",
        "-ltorch_python",
    ]

    # Platform-specific compiler & linker flags
    if sys.platform == "darwin":
        cmd.extend(["-mcpu=apple-m3", "-undefined", "dynamic_lookup"])
    else:
        cmd.extend(["-march=armv8-a+simd"])

    cmd.extend([CPP_SOURCE, "-o", SO_OUTPUT])

    if verbose:
        print("Compiling NEON kernels with command:", " ".join(cmd))

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Failed to compile Axiom NEON kernels:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        )

    return SO_OUTPUT


def load_neon_module() -> Optional[Any]:
    """
    Compiles (if needed) and dynamically imports the native NEON extension.
    Returns the module or None if compilation/import fails or not on ARM.
    """
    if not IS_ARM:
        return None

    try:
        so_path = compile_neon_extension(force=False)
        spec = importlib.util.spec_from_file_location("axiom_neon_kernels", so_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"Notice: Native NEON kernel load fallback: {e}")
        return None


if __name__ == "__main__":
    if not IS_ARM:
        print(f"Skipping NEON kernel build: Host architecture ({platform.machine()}) is not ARM64/aarch64.")
    else:
        print(f"Building Axiom-LM Apple Silicon Kernels...")
        so = compile_neon_extension(force=True, verbose=True)
        print(f"Successfully compiled: {so}")
        mod = load_neon_module()
        print("Exported module symbols:", dir(mod))
