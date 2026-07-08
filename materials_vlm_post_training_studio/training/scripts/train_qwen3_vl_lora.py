import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training.src.data.jsonl import read_jsonl, resolve_project_path
from training.src.models import Qwen3VLWrapper, Qwen3VLWrapperConfig
from training.src.utils.config import load_yaml, merge_cli_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT entry point for Qwen3-VL.")
    parser.add_argument("--config", default="training/configs/qwen3_vl_lora.yaml")
    parser.add_argument("--train_file")
    parser.add_argument("--val_file")
    parser.add_argument("--output_dir")
    parser.add_argument("--base_model")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Validate config/data without loading model weights.")
    parser.add_argument("--allow_download", action="store_true", help="Allow HF downloads. Off by default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config, PROJECT_ROOT)
    config = merge_cli_overrides(
        load_yaml(config_path),
        {
            "train_file": args.train_file,
            "val_file": args.val_file,
            "output_dir": args.output_dir,
            "model_name": args.base_model,
        },
    )
    if args.qlora:
        config["use_4bit"] = True

    train_file = resolve_project_path(config["train_file"], PROJECT_ROOT)
    val_file = resolve_project_path(config["val_file"], PROJECT_ROOT)
    output_dir = resolve_project_path(config["output_dir"], PROJECT_ROOT)
    train_rows = read_jsonl(train_file)
    val_rows = read_jsonl(val_file)

    wrapper = Qwen3VLWrapper(
        Qwen3VLWrapperConfig(
            model_name_or_path=config["model_name"],
            use_4bit=bool(config.get("use_4bit", False)),
            local_files_only=not args.allow_download,
            torch_dtype=config.get("torch_dtype", "auto"),
            attn_implementation=config.get("attn_implementation"),
        )
    )
    preview = [wrapper.preprocess_for_training(row, config.get("image_root")) for row in train_rows[:2]]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_ok",
                    "model_name": config["model_name"],
                    "local_files_only": not args.allow_download,
                    "use_4bit": bool(config.get("use_4bit", False)),
                    "train_examples": len(train_rows),
                    "val_examples": len(val_rows),
                    "preview": preview,
                    "note": "No model weights were loaded or downloaded.",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Real SFT requires transformers, peft, accelerate, torch, pillow, pyyaml, and qwen-vl-utils.") from exc

    wrapper.load_model()
    assert wrapper.model is not None
    if config.get("use_4bit", False):
        wrapper.model = prepare_model_for_kbit_training(wrapper.model)
    lora_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    wrapper.model = get_peft_model(wrapper.model, lora_config)

    raise NotImplementedError(
        "Qwen3-VL model-specific multimodal collator is intentionally left as a narrow TODO. "
        "The wrapper/config/local-files loading path is ready; implement token-label masking after confirming "
        "the exact Qwen3-VL processor output for your installed Transformers version."
    )


if __name__ == "__main__":
    main()

