.PHONY: check
check: lint typecheck test-rust test-python test-integration
	@echo "--- ALL CHECKS PASSED ---"

.PHONY: lint
lint:
	source .venv/bin/activate && cargo clippy --all-targets -- -D warnings
	source .venv/bin/activate && ruff check python/ tests/
	source .venv/bin/activate && ruff format --check python/ tests/

.PHONY: typecheck
typecheck:
	source .venv/bin/activate && mypy python/pyjit/ --strict

.PHONY: test-rust
test-rust:
	source .venv/bin/activate && cargo test --all

.PHONY: test-python
test-python:
	source .venv/bin/activate && python -m pytest tests/ -x -v --tb=short --ignore=tests/integration --ignore=tests/benchmarks

.PHONY: test-integration
test-integration:
	source .venv/bin/activate && python -m pytest tests/integration/ -x -v --tb=short

.PHONY: bench
bench:
	source .venv/bin/activate && python -m pytest tests/benchmarks/ -x -v --tb=short --benchmark-only

.PHONY: build
build:
	source .venv/bin/activate && maturin develop --release

.PHONY: build-debug
build-debug:
	source .venv/bin/activate && maturin develop

.PHONY: clean
clean:
	cargo clean
	rm -rf target/ dist/ *.egg-info .pytest_cache
