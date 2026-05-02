# src/common/data_manager.py
import pickle
import os
from typing import Any

class DataManager:
    @staticmethod
    def save(data: Any, path: str):
        """Save arbitrary data in pickle format."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[DataManager] Saved to {path}")

    @staticmethod
    def load(path: str) -> Any:
        """Load data from a pickle file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"[DataManager] Loaded from {path}")
        return data
