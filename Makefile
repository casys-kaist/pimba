format:
	uv run ruff check --fix --select=I001
	uv run ruff format
	git ls-files '*.c' '*.cpp' '*.cc' '*.h' '*.hh' '*.hpp' | xargs uv run clang-format -i
