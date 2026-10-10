"""Run directly from the checkout: .venv/bin/python app.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from arrow_solver.app import main

if __name__ == "__main__":
    main()

