import os
import sys
from pathlib import Path


def main():
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.load_data import load_dataset
    statistics = load_dataset("WikiCS", statistics=True)
    for key, value in statistics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
