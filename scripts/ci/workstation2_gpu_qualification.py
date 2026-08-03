"""Qualify a self-hosted CUDA/Warp runner with deterministic shared-memory IO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import warp as wp


@wp.kernel
def _saxpy_kernel(
    x: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.float32),
    output: wp.array(dtype=wp.float32),
    alpha: float,
) -> None:
    index = wp.tid()
    output[index] = alpha * x[index] + y[index]


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _throughput_gb_s(elements: int, iterations: int, seconds: float) -> float:
    transferred_bytes = elements * iterations * 3 * np.dtype(np.float32).itemsize
    return transferred_bytes / seconds / 1e9


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic Torch and Warp CUDA qualification."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--elements", type=int, default=4_000_000)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--maximum-error", type=float, default=2e-6)
    return parser


def run_qualification(args: argparse.Namespace) -> dict[str, Any]:
    if args.elements < 1:
        raise ValueError("elements must be positive")
    if args.iterations < 1 or args.warmup_iterations < 0:
        raise ValueError("iteration counts are invalid")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a CUDA device")

    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    torch_device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(torch_device)

    wp.init()
    cuda_devices = wp.get_cuda_devices()
    if not cuda_devices:
        raise RuntimeError("Warp cannot see a CUDA device")
    warp_device = cuda_devices[0]
    wp.set_device(warp_device)

    index = torch.arange(args.elements, device=torch_device, dtype=torch.float32)
    x = torch.sin(index * 0.001)
    y = torch.cos(index * 0.0007)
    output = torch.empty_like(x)
    reference = torch.add(y, x, alpha=args.alpha)

    wp_x = wp.from_torch(x, dtype=wp.float32)
    wp_y = wp.from_torch(y, dtype=wp.float32)
    wp_output = wp.from_torch(output, dtype=wp.float32)

    for _ in range(args.warmup_iterations):
        wp.launch(
            _saxpy_kernel,
            dim=args.elements,
            inputs=[wp_x, wp_y, wp_output, args.alpha],
            device=warp_device,
        )
    wp.synchronize()

    started = time.perf_counter()
    for _ in range(args.iterations):
        wp.launch(
            _saxpy_kernel,
            dim=args.elements,
            inputs=[wp_x, wp_y, wp_output, args.alpha],
            device=warp_device,
        )
    wp.synchronize()
    warp_seconds = time.perf_counter() - started

    maximum_error = float(torch.max(torch.abs(output - reference)).item())
    first_output = output.detach().cpu().numpy().copy()
    first_sha256 = _sha256_array(first_output)

    output.fill_(float("nan"))
    wp.launch(
        _saxpy_kernel,
        dim=args.elements,
        inputs=[wp_x, wp_y, wp_output, args.alpha],
        device=warp_device,
    )
    wp.synchronize()
    second_output = output.detach().cpu().numpy().copy()
    second_sha256 = _sha256_array(second_output)

    torch_output = torch.empty_like(x)
    for _ in range(args.warmup_iterations):
        torch.add(y, x, alpha=args.alpha, out=torch_output)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.iterations):
        torch.add(y, x, alpha=args.alpha, out=torch_output)
    torch.cuda.synchronize()
    torch_seconds = time.perf_counter() - started
    torch_maximum_error = float(torch.max(torch.abs(torch_output - reference)).item())

    passed = bool(
        maximum_error <= args.maximum_error
        and torch_maximum_error <= args.maximum_error
        and first_sha256 == second_sha256
    )
    result = {
        "schema_version": 1,
        "artifact_kind": "Causal4DWorkstation2GpuQualification",
        "passed": passed,
        "runner": {
            "name": os.environ.get("RUNNER_NAME"),
            "os": os.environ.get("RUNNER_OS"),
            "architecture": os.environ.get("RUNNER_ARCH"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "platform": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "gpu": {
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "warp_version": wp.__version__,
            "warp_cuda_devices": [str(device) for device in cuda_devices],
        },
        "configuration": {
            "elements": args.elements,
            "iterations": args.iterations,
            "warmup_iterations": args.warmup_iterations,
            "alpha": args.alpha,
            "maximum_error": args.maximum_error,
        },
        "correctness": {
            "warp_maximum_absolute_error": maximum_error,
            "torch_maximum_absolute_error": torch_maximum_error,
            "first_output_sha256": first_sha256,
            "second_output_sha256": second_sha256,
            "repeat_is_bit_identical": first_sha256 == second_sha256,
        },
        "performance": {
            "warp_seconds": warp_seconds,
            "warp_effective_gb_s": _throughput_gb_s(
                args.elements, args.iterations, warp_seconds
            ),
            "torch_seconds": torch_seconds,
            "torch_effective_gb_s": _throughput_gb_s(
                args.elements, args.iterations, torch_seconds
            ),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "claim_boundary": (
            "Hardware and numerical-runtime qualification only; this is not an "
            "accuracy, calibration, physical-prediction, or intervention result."
        ),
    }
    if not passed:
        raise RuntimeError(f"GPU qualification failed: {result['correctness']}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_qualification(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
