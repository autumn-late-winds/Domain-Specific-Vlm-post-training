import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training.src.data.jsonl import read_jsonl, resolve_project_path
from training.src.models import Qwen3VLWrapper, Qwen3VLWrapperConfig
from training.src.utils.config import load_yaml, merge_cli_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO entry point for Qwen3-VL preference data.")
    parser.add_argument("--config", default="training/configs/qwen3_vl_dpo.yaml")
    parser.add_argument("--train_file")
    parser.add_argument("--output_dir")
    parser.add_argument("--base_model")
    parser.add_argument("--dry_run", action="store_true", help="Validate preference formatting without loading model weights.")
    parser.add_argument("--allow_download", action="store_true", help="Allow HF downloads. Off by default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = merge_cli_overrides(
        load_yaml(resolve_project_path(args.config, PROJECT_ROOT)),
        {"train_file": args.train_file, "output_dir": args.output_dir, "model_name": args.base_model},
    )
    rows = read_jsonl(resolve_project_path(config["train_file"], PROJECT_ROOT))
    wrapper = Qwen3VLWrapper(
        Qwen3VLWrapperConfig(
            model_name_or_path=config["model_name"],
            use_4bit=bool(config.get("use_4bit", False)),
            local_files_only=not args.allow_download,
            torch_dtype=config.get("torch_dtype", "auto"),
        )
    )
    preview = []
    for row in rows[:2]:
        preview.append(
            {
                "id": row["id"],
                "prompt_messages": wrapper.build_messages(row["image"], row["question"]),
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_ok",
                    "model_name": config["model_name"],
                    "local_files_only": not args.allow_download,
                    "preference_examples": len(rows),
                    "preview": preview,
                    "note": "No model weights were loaded or downloaded.",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    try:
        from trl import DPOTrainer
    except ImportError as exc:
        raise RuntimeError("Real DPO requires trl plus the real Qwen3-VL training dependency stack.") from exc

    wrapper.load_model()
    raise NotImplementedError(
        "Qwen3-VL DPO requires a model-specific multimodal DPO collator. The preference-data interface is ready; "
        "finish after confirming the installed TRL version's multimodal DPO API."
    )


if __name__ == "__main__":
    main()

