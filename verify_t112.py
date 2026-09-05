#!/usr/bin/env python3
"""Proofs 2-4: topology, golden checksum, fail-closed. Exit 0 or assert."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t112_zero_gemm_bench import (  # noqa: E402
    DEG,
    HOPS,
    LUT_BYTES,
    N,
    build_circulant_lut,
    hop_loop,
    hop_loop_failclosed,
)

GOLDEN_NODE = 4972
GOLDEN_ACC = 158148294537
GOLDEN_SHA256 = "9d9bc34b4df51e88d71d280bb0547e8b9a117c5873ab3582792dd6a7059bec68"


def verify_layout(lut: np.ndarray) -> None:
    assert lut.dtype == np.uint16
    assert lut.nbytes == LUT_BYTES == 75936
    assert lut.nbytes <= 256 * 1024


def verify_topological_invariants(lut: np.ndarray, n: int = N, degree: int = DEG) -> None:
    assert lut.shape == (n, degree)
    assert int(lut.min()) >= 0 and int(lut.max()) < n
    s = np.sort(lut, axis=1)
    assert np.all(s[:, 1:] > s[:, :-1]), "duplicate out-edges"
    in_deg = np.bincount(lut.ravel().astype(np.int32), minlength=n)
    assert in_deg.shape == (n,)
    assert np.all(in_deg == degree), "in-degree != out-degree"
    seen = np.zeros(n, dtype=np.uint8)
    stack = [0]
    while stack:
        cur = stack.pop()
        if seen[cur]:
            continue
        seen[cur] = 1
        stack.extend(int(x) for x in lut[cur])
    assert int(seen.sum()) == n, f"not connected: {int(seen.sum())}/{n}"


def verify_golden_determinism(lut, seed=7, hops=HOPS) -> str:
    final_node, acc = hop_loop(lut, hops, seed)
    dump = f"{int(final_node)}:{int(acc)}".encode("utf-8")
    actual = hashlib.sha256(dump).hexdigest()
    assert int(final_node) == GOLDEN_NODE
    assert int(acc) == GOLDEN_ACC
    assert actual == GOLDEN_SHA256, f"expected {GOLDEN_SHA256} got {actual}"
    return actual


def verify_fail_closed_guarantee(lut, poison_indices=(6328, 65535, 99999)) -> None:
    poison_hop = 500
    for poison in poison_indices:
        halted_hop, error_code = hop_loop_failclosed(
            lut, 10_000, 7, poison_hop, int(poison)
        )
        assert error_code == 1, f"missed poison {poison}"
        assert halted_hop == poison_hop, f"leaked: halt {halted_hop} expected {poison_hop}"
    hops, err = hop_loop_failclosed(lut, 10_000, 7, 10_000_000, -1)
    assert err == 0 and hops == 10_000


def main() -> None:
    lut = build_circulant_lut()
    verify_layout(lut)
    print("[+] layout 75936 B uint16")
    verify_topological_invariants(lut)
    print("[+] topology closed 6-regular connected")
    h = verify_golden_determinism(lut)
    print("[+] golden", h)
    verify_fail_closed_guarantee(lut)
    print("[+] fail-closed 6328/65535/99999 halt at hop 500")
    print("[+] ALL PYTHON PROOFS PASSED")


if __name__ == "__main__":
    main()
