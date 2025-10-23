"""
Preprocessor registry so additional datasets can plug into Orchid Ranker.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

__all__ = [
    "BasePreprocessor",
    "PreprocessorConfig",
    "register_preprocessor",
    "get_preprocessor",
    "list_preprocessors",
]


@dataclass
class PreprocessorConfig:
    """Common parameters forwarded to preprocessors."""

    base_path: str
    output_path: str
    seed: int = 42
    n_users: Optional[int] = None
    extra: Optional[dict] = None


class BasePreprocessor(ABC):
    """Base class for dataset-specific preprocessing pipelines."""

    name: str = "base"

    @abstractmethod
    def run(self, cfg: PreprocessorConfig) -> None:
        """Execute preprocessing and save artefacts to ``cfg.output_path``."""


_REGISTRY: Dict[str, Callable[[], BasePreprocessor]] = {}


def register_preprocessor(name: str):
    """Decorator to register a preprocessor under a given name."""

    def _decorator(cls: Type[BasePreprocessor]):
        _REGISTRY[name.lower()] = cls
        return cls

    return _decorator


def get_preprocessor(name: str) -> BasePreprocessor:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown preprocessor '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[key]()


def list_preprocessors() -> Dict[str, Callable[[], BasePreprocessor]]:
    return dict(_REGISTRY)

