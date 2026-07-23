from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config loading requires pyyaml.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def merge_cli_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged

