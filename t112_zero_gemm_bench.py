#!/usr/bin/env python3
"""Bare-metal-style T112 LUT hop vs dense GEMM. Prints JSON telemetry."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
from numba import njit

N = 6328  # T_112 = 112 * 113 / 2
DEG = 6
LUT_BYTES = N * DEG * 2  # uint16
HOPS = 50_000_000
GEMM_REPS = 200

OUT = Path(__file__).with_name("t112_zero_gemm_telemetry.json")


def build_circulant_lut() -> np.ndarray:
    """6-regular circulant graph on N nodes. Offsets ±1, ±2, ±3."""
    offsets = np.array([1, 2, 3, N - 1, N - 2, N - 3], dtype=np.int32)
    nodes = np.arange(N, dtype=np.int32)
    lut = ((nodes[:, None] + offsets[None, :]) % N).astype(np.uint16)
    return lut


@njit(cache=True)
def hop_loop(lut, hops, seed):
    node = np.uint16(seed % N)
    acc = np.uint64(0)
    d = lut.shape[1]
    for i in range(hops):
        direction = (node + i) % d
        node = lut[node, direction]
        acc += np.uint64(node)
    return np.int64(node), np.int64(acc)


@njit(cache=True)
def hop_loop_failclosed(lut, hops, seed, poison_at, poison_val=-1):
    """Same walk; at poison_at inject poison_val (default N+1) and halt."""
    node = np.uint16(seed % N)
    n = lut.shape[0]
    d = lut.shape[1]
    for i in range(hops):
        if i == poison_at:
            nxt = n + 1 if poison_val < 0 else poison_val
            if nxt >= n:
                return i, 1
        direction = (node + i) % d
        nxt = int(lut[node, direction])
        if nxt >= n:
            return i, 1
        node = np.uint16(nxt)
    return hops, 0


def median_s(fn, repeats=7, warmup=2):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2], samples


def gemm_once(a, b):
    return a @ b


def main():
    lut = build_circulant_lut()
    assert lut.nbytes == LUT_BYTES
    assert lut.max() < N

    hop_loop(lut, 10_000, 7)
    hop_loop_failclosed(lut, 10_000, 7, 3)

    hop_s, hop_samples = median_s(lambda: hop_loop(lut, HOPS, 7))
    ns_per_hop = hop_s * 1e9 / HOPS
    hops_per_s = HOPS / hop_s

    def poison():
        hop_loop_failclosed(lut, 1_000_000, 7, 100)

    poison_s, _ = median_s(poison, repeats=11, warmup=5)
    halt_detect_ns = poison_s * 1e9

    gemm_rows = []
    rng = np.random.default_rng(0)
    for dim in (64, 256, 512, 1024):
        a = rng.standard_normal((dim, dim), dtype=np.float32)
        b = rng.standard_normal((dim, dim), dtype=np.float32)
        gemm_once(a, b)
        reps = max(8, GEMM_REPS // max(1, dim // 64))
        s, _ = median_s(lambda: gemm_once(a, b), repeats=reps, warmup=3)
        flops = 2.0 * dim * dim * dim
        gemm_rows.append(
            {
                "dim": dim,
                "dtype": "float32",
                "bytes_ab": int(2 * dim * dim * 4),
                "median_s": s,
                "ns": s * 1e9,
                "gflops": (flops / s) / 1e9,
                "hops_in_one_gemm": hops_per_s * s,
            }
        )

    uname = platform.uname()
    report = {
        "host": {
            "system": uname.system,
            "machine": uname.machine,
            "processor": uname.processor,
            "python": platform.python_version(),
        },
        "lattice": {
            "N": N,
            "degree": DEG,
            "dtype": "uint16",
            "lut_bytes": LUT_BYTES,
            "lut_kib": LUT_BYTES / 1024.0,
            "construction": "circulant offsets {+/-1,+/-2,+/-3} mod N",
        },
        "hops": {
            "count": HOPS,
            "median_s": hop_s,
            "ns_per_hop": ns_per_hop,
            "hops_per_s": hops_per_s,
            "samples_s": hop_samples,
        },
        "fail_closed": {
            "poison_after_hops": 100,
            "median_detect_s": poison_s,
            "detect_ns": halt_detect_ns,
        },
        "gemm": gemm_rows,
        "note": "Hops are integer LUT walks, not language-model tokens. GEMM is dense float32 matmul of the stated order.",
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
