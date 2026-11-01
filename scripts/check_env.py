"""Environment & Hardware Diagnostics for FineTuneOps.

Audits:
- Operating system and platform architecture
- Python version and installed core packages
- CPU and system RAM availability
- NVIDIA GPU presence, driver version, VRAM, and CUDA compute capability
- PyTorch CUDA bindings and compatibility
- Storage / disk volume availability
- Node.js & container tools
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict


def get_system_ram() -> Dict[str, float]:
    """Retrieve total and available physical RAM in GB across platforms."""
    if platform.system() == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return {
                "total_gb": round(stat.ullTotalPhys / (1024**3), 2),
                "available_gb": round(stat.ullAvailPhys / (1024**3), 2),
            }
    # Fallback to sysconf if available (Linux/POSIX)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        return {
            "total_gb": round((pages * page_size) / (1024**3), 2),
            "available_gb": round((avail_pages * page_size) / (1024**3), 2),
        }
    except Exception:
        return {"total_gb": -1.0, "available_gb": -1.0}


def get_disk_storage() -> Dict[str, Dict[str, float]]:
    """Inspect disk capacity across primary volumes."""
    drives = ["C:\\", "E:\\", "F:\\"] if platform.system() == "Windows" else ["/"]
    storage: Dict[str, Dict[str, float]] = {}
    for d in drives:
        if os.path.exists(d):
            try:
                t, u, f = shutil.disk_usage(d)
                storage[d] = {
                    "total_gb": round(t / (1024**3), 2),
                    "used_gb": round(u / (1024**3), 2),
                    "free_gb": round(f / (1024**3), 2),
                }
            except Exception:
                pass
    return storage


def get_nvidia_smi() -> Dict[str, Any]:
    """Query nvidia-smi for hardware GPU device and memory."""
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        line = res.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        return {
            "gpu_detected": True,
            "name": parts[0],
            "driver_version": parts[1],
            "total_vram_mb": float(parts[2]),
            "free_vram_mb": float(parts[3]),
            "used_vram_mb": float(parts[4]),
        }
    except Exception as e:
        return {"gpu_detected": False, "error": str(e)}


def check_pytorch_cuda() -> Dict[str, Any]:
    """Inspect PyTorch installation and its CUDA linkage."""
    try:
        import torch

        cuda_avail = torch.cuda.is_available()
        return {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": cuda_avail,
            "device_count": torch.cuda.device_count() if cuda_avail else 0,
            "device_name": torch.cuda.get_device_name(0) if cuda_avail else None,
            "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        }
    except ImportError:
        return {"installed": False}


def check_cli(cmd: str) -> str | None:
    """Return version string of a CLI command if found in PATH."""
    try:
        res = subprocess.run([cmd, "--version"], capture_output=True, text=True, check=True)
        return res.stdout.strip().split("\n")[0]
    except Exception:
        return None


def run_assessment() -> None:
    """Execute complete environment assessment and print formatted summary."""
    print("=" * 70)
    print("           FINETUNEOPS - ENVIRONMENT & HARDWARE ASSESSMENT        ")
    print("=" * 70)

    # OS & Architecture
    print(f"\n[OS & SYSTEM]")
    print(f"  Platform         : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"  Architecture     : {platform.machine()}")
    print(f"  Python Version   : {sys.version.split()[0]} ({sys.executable})")

    # RAM
    ram = get_system_ram()
    print(f"\n[SYSTEM MEMORY]")
    print(f"  Total RAM        : {ram['total_gb']} GB")
    print(f"  Available RAM    : {ram['available_gb']} GB")

    # Disk
    print(f"\n[STORAGE VOLUMES]")
    storage = get_disk_storage()
    for drive, info in storage.items():
        print(f"  Drive {drive:<8}  : Total={info['total_gb']} GB | Free={info['free_gb']} GB | Used={info['used_gb']} GB")

    # NVIDIA GPU
    print(f"\n[NVIDIA HARDWARE]")
    gpu = get_nvidia_smi()
    if gpu.get("gpu_detected"):
        print(f"  GPU Name         : {gpu['name']}")
        print(f"  Driver Version   : {gpu['driver_version']}")
        print(f"  VRAM Total       : {gpu['total_vram_mb']} MB ({gpu['total_vram_mb'] / 1024:.2f} GB)")
        print(f"  VRAM Free        : {gpu['free_vram_mb']} MB ({gpu['free_vram_mb'] / 1024:.2f} GB)")
    else:
        print("  No NVIDIA GPU detected via nvidia-smi.")

    # PyTorch CUDA
    print(f"\n[PYTORCH & CUDA INTEGRATION]")
    torch_info = check_pytorch_cuda()
    if torch_info["installed"]:
        print(f"  PyTorch Version  : {torch_info['version']}")
        print(f"  CUDA Available   : {torch_info['cuda_available']}")
        if torch_info["cuda_available"]:
            print(f"  Device Name      : {torch_info['device_name']}")
            print(f"  PyTorch CUDA     : {torch_info['cuda_version']}")
        else:
            print("  Notice: PyTorch is currently CPU-only (+cpu).")
    else:
        print("  PyTorch is not installed.")

    # Developer Tools
    print(f"\n[DEV & CONTAINER TOOLS]")
    for tool in ["git", "docker", "docker-compose", "node", "npm"]:
        v = check_cli(tool)
        status = v if v else "Not found in PATH"
        print(f"  {tool:<16} : {status}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_assessment()
