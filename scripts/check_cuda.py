##!/usr/bin/env python
import argparse
import torch
from rtk.repl import prepare_console


# Check GPU Drivers and CUDA Version
def check_gpu_driver():
    if torch.cuda.is_available():
        current_device = args.device
        # driver_version = torch.cuda.get_dr
        cuda_version = torch.version.cuda
        console.print(f"Current device: '{current_device}'")
        # console.print(f"GPU Driver Version: '{driver_version}'")
        console.print(f"CUDA Version: '{cuda_version}'")
    else:
        console.print("No GPU available.")


# Check GPU memory i.e how much memory is there, how much is free
def check_gpu_memory():
    if torch.cuda.is_available():
        current_device = args.device
        gpu = torch.cuda.get_device_properties(current_device)
        console.print(f"GPU Name: '{gpu.name}'")
        console.print(f"GPU Memory Total: '{gpu.total_memory / 1024**2} MB'")
        console.print(
            f"GPU Memory Free: '{torch.cuda.memory_allocated(current_device) / 1024**2} MB'"
        )
        console.print(
            f"GPU Memory Used: '{torch.cuda.memory_reserved(current_device) / 1024**2} MB'"
        )
    else:
        console.print("No GPU available.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check CUDA and GPU drivers.")
    parser.add_argument("--device", help="The device to check", default="cuda:0")
    args = parser.parse_args()
    ws, console = prepare_console()
    console.print(f"Torch version: '{torch.__version__}'")

    if not torch.cuda.is_available():
        console.print("CUDA driver is not installed.")
    else:
        console.print("CUDA driver is installed.")

    if torch.backends.cudnn.is_available():
        console.print("cuDNN is installed.")
    else:
        console.print("cuDNN is not installed.")

    check_gpu_driver()
    check_gpu_memory()
