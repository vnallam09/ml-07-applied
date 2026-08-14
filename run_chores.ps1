# ============================================================
# run_chores.ps1 - update dependencies, lint, test, and build
# ============================================================

# PowerShell script to automate the common workflow.
# If on mac or Linux, you can convert this to a bash script.
# The commands are similar.

# Run in a PowerShell terminal from the project root directory:
# .\run_chores.ps1

# Clear the terminal for better readability
Clear-Host

uv self update
uv python pin 3.14
uv lock --upgrade
uv sync --extra dev --extra docs --upgrade

uvx pre-commit install
uvx pre-commit autoupdate

git add -A
uvx pre-commit run --all-files
# repeat if changes were made
uvx pre-commit run --all-files

# run common chores
uv run ruff format .
uv run ruff check . --fix
uv run python -m pyright
uv run python -m pytest
uv run python -m zensical build

Write-Host "All commands executed successfully. "
Write-Host "Run the example app to verify .venv/ is working correctly."
Write-Host "We work in the notebooks/ folder in this course."
