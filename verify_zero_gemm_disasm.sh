#!/usr/bin/env bash
# Proof 5: hop inner loop has no FP/FMA opcodes.
set -euo pipefail
cd "$(dirname "$0")"
gcc -O3 -march=native -std=c11 bench_l2_graph.c -o bench_l2_graph
DISASM=$(objdump -d -M intel bench_l2_graph | awk '/<run_hops>:/{p=1} p&&/^$/{if(++b>1) exit} p')
echo "$DISASM"
FORBIDDEN=(vfmadd vmulss vaddss mulss addss movss fadd fmul vfmadd231ss)
for op in "${FORBIDDEN[@]}"; do
  if echo "$DISASM" | grep -E -q "(^|[[:space:]])${op}[[:space:]]"; then
    echo "[!] FP opcode in hop loop: $op"
    exit 1
  fi
done
echo "[+] AUDIT PASSED: run_hops is integer-only in the dump"
