from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class VLMWrapper(ABC):
    @abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_messages(self, image_path: str | Path, question: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def generate(self, image_path: str | Path, question: str, max_new_tokens: int = 256) -> str:
        raise NotImplementedError

    @abstractmethod
    def save_adapter(self, output_dir: str | Path) -> None:
        raise NotImplementedError

