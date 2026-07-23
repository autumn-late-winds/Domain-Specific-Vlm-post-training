import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training.src.models import Qwen3VLWrapper, Qwen3VLWrapperConfig


def test_qwen3_vl_messages_are_multimodal():
    wrapper = Qwen3VLWrapper(Qwen3VLWrapperConfig(model_name_or_path="Qwen/Qwen3-VL-2B-Instruct"))
    messages = wrapper.build_messages("data/demo/images/sem_001.svg", "Describe the morphology.")
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "image"
    assert messages[0]["content"][1]["type"] == "text"


def test_qwen3_vl_default_is_local_files_only():
    config = Qwen3VLWrapperConfig(model_name_or_path="Qwen/Qwen3-VL-2B-Instruct")
    assert config.local_files_only is True


def test_qwen3_vl_lora_dry_run_does_not_load_model():
    result = subprocess.run(
        [
            sys.executable,
            "training/scripts/train_qwen3_vl_lora.py",
            "--config",
            "training/configs/qwen3_vl_lora.yaml",
            "--dry_run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "dry_run_ok" in result.stdout
    assert "No model weights were loaded or downloaded" in result.stdout


def test_qwen3_vl_dpo_dry_run_does_not_load_model():
    result = subprocess.run(
        [
            sys.executable,
            "training/scripts/train_qwen3_vl_dpo.py",
            "--config",
            "training/configs/qwen3_vl_dpo.yaml",
            "--dry_run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "dry_run_ok" in result.stdout
    assert "preference_examples" in result.stdout
