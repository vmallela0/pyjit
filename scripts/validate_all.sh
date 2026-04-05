#!/usr/bin/env bash
set -euo pipefail

echo "=== Building with maturin ==="
maturin develop

echo ""
echo "=== Running smoke test ==="
python scripts/smoke_test.py

echo ""
echo "=== Running make check ==="
make check

echo ""
echo "All validation passed!"
