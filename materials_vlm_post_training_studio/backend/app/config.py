from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DEMO_DIR = DATA_DIR / "demo"
OUTPUT_DIR = ROOT_DIR / "outputs" / "experiments"

DEMO_DATASET_ID = "demo_materials_vlm"
DEMO_TRAIN_FILE = DEMO_DIR / "sft_train.jsonl"
DEMO_VAL_FILE = DEMO_DIR / "sft_val.jsonl"
DEMO_DPO_FILE = DEMO_DIR / "dpo_train.jsonl"

