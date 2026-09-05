# t112-l2-lattice

**74.16 KiB. Degree 6. Integer hops. No GEMM.**

A 6-regular graph on N = T_112 = 6328 nodes. The whole adjacency table is `uint16[6328][6]` = **75,936 bytes**. It fits in a **256 KiB** Kaby Lake L2 (29%). It is not a language model.

| | |
|---|---|
| Paper | [`PAPER.md`](PAPER.md) · [doi:10.5281/zenodo.22406010](https://doi.org/10.5281/zenodo.22406010) |
| Table | [UltranetCommand/t112-l2-lattice](https://huggingface.co/datasets/UltranetCommand/t112-l2-lattice) (**dataset**, not a model) |

```text
python verify_t112.py
python t112_zero_gemm_bench.py
```

Linux hop + disasm:

```text
gcc -O3 -march=native -std=c11 bench_l2_graph.c -o bench_l2_graph
bash verify_zero_gemm_disasm.sh
```

## Two-socket telemetry

| | i7-7700 (Neo) | i7-14700F |
|---|---|---|
| L2 | 256 KiB / core | 2 MiB / P-core |
| Table | 74.16 KiB (28.9%) | 74.16 KiB (3.6%) |
| Hop | **9.76 ns** / 102M hops/s | **5.75 ns** / 174M hops/s |
| Halt | 1.5 µs | 800 ns |
| 1024×1024 f32 GEMM | 7.48 ms = **767k hops** | 3.56 ms = 619k hops |
| Golden SHA-256 | `9d9bc34b…bec68` | **identical** |

JSON: `t112_zero_gemm_telemetry_neo.json`, `t112_zero_gemm_telemetry_local.json`

## Proofs

| | |
|---|---|
| `static_assert` | `sizeof(Node)==12`, 75936 ≤ 256 KiB |
| `verify_t112.py` | 6-regular, connected, golden checksum, fail-closed |
| `verify_zero_gemm_disasm.sh` | `run_hops` has no FP/FMA opcodes |

## Requirements

```text
pip install numpy numba
```

Hops are not tokens. GEMM is vendor BLAS, not a naive triple loop.

Apache-2.0. Matt Gibson / Crimson OS.
