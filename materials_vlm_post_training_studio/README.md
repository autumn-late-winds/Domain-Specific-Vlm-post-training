# Materials VLM Post Training Studio

A runnable first-round MVP for demonstrating domain-specific post-training workflows for vision-language models in materials science.

This version intentionally focuses on a fast portfolio-ready loop:

- Synthetic demo dataset with SEM-like, XRD, Raman, and performance-curve figures.
- FastAPI backend with dataset validation, mock training, mock evaluation, inference, and report APIs.
- React + Vite dashboard for dataset preview, training, evaluation, playground, and report review.
- Deterministic mock VLM behavior so the project opens without downloading large models or needing a GPU.

## Why This Matters

Materials science VLM applications need answers that are specific, grounded, and cautious. A general model may describe an image generically or hallucinate composition and performance. This project shows the engineering skeleton for post-training and evaluation before connecting real LoRA, QLoRA, and DPO training.

## Architecture

```text
backend/
  app/
    routers/        FastAPI endpoints
    services/       dataset, mock VLM, mock training, evaluation, report logic
frontend/
  src/              React dashboard
data/demo/
  images/           synthetic SVG figures
  sft_train.jsonl   supervised fine-tuning examples
  sft_val.jsonl     validation examples
  dpo_train.jsonl   preference examples for later DPO work
outputs/
  experiments/      generated logs, summaries, reports
training/
  scripts/          CLI entry points
  configs/          mock config now, real configs later
```

## Run Locally

Backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Run the Mock Pipeline from CLI

```bash
python training/scripts/run_mock_pipeline.py --epochs 3
```

This writes:

- `outputs/experiments/{experiment_id}/training_logs.json`
- `outputs/experiments/{experiment_id}/experiment_summary.json`
- `outputs/experiments/{experiment_id}/evaluation_base.json`
- `outputs/experiments/{experiment_id}/evaluation_finetuned.json`
- `outputs/experiments/{experiment_id}/comparison_report.json`

## Qwen3-VL Interface, No Download by Default

The project now includes Qwen3-VL LoRA/QLoRA and DPO entry points, but they are interface-first. They do not download model files by default and they do not affect mock mode.

Dry-run SFT formatting:

```bash
python training/scripts/train_qwen3_vl_lora.py --config training/configs/qwen3_vl_lora.yaml --dry_run
```

Dry-run QLoRA formatting:

```bash
python training/scripts/train_qwen3_vl_lora.py --config training/configs/qwen3_vl_qlora.yaml --qlora --dry_run
```

Dry-run DPO formatting:

```bash
python training/scripts/train_qwen3_vl_dpo.py --config training/configs/qwen3_vl_dpo.yaml --dry_run
```

Real mode expects Qwen3-VL files to already exist in the local Hugging Face cache or at a local path. The wrapper uses `local_files_only=True` unless you explicitly pass `--allow_download`.

Optional real-VLM dependencies are listed in:

```bash
training/requirements-real-vlm.txt
```

The remaining real-training TODO is the Qwen3-VL-specific multimodal collator and label masking, which should be finalized against the exact installed `transformers` and `trl` versions.

## Dataset Format

SFT JSONL:

```json
{"id":"sem_001","image":"data/demo/images/sem_001.svg","task_type":"sem_description","question":"What morphology is shown?","answer":"The image shows porous nanosheets.","metadata":{"modality":"SEM"}}
```

DPO JSONL:

```json
{"id":"pref_001","image":"data/demo/images/sem_001.svg","question":"Describe morphology.","chosen":"Specific grounded answer.","rejected":"Generic answer.","metadata":{"modality":"SEM"}}
```

## API Highlights

- `GET /health`
- `GET /datasets`
- `GET /datasets/demo_materials_vlm/examples`
- `POST /datasets/demo_materials_vlm/validate`
- `POST /training/start`
- `POST /evaluation/run`
- `POST /inference/query`
- `GET /reports/{experiment_id}/markdown`
- `GET /reports/{experiment_id}/json`

## Tests

```bash
cd backend
pytest
```

## Docker Demo

```bash
docker compose up --build
```

Frontend: `http://localhost:5173`

Backend: `http://localhost:8000`

## Current Working Scope

- Demo dataset loads locally.
- Dataset schema validation and task/modality statistics work.
- Mock training creates experiment folders and logs.
- Mock evaluation compares deterministic base and fine-tuned responses.
- Dashboard shows dataset examples, starts mock training, runs evaluation, shows before/after examples, and supports playground inference.
- JSON and Markdown reports are generated.

## Limitations

- Mock mode does not train real model weights.
- Synthetic SVG figures validate the workflow but are not a scientific benchmark.
- Metrics are lightweight rule-based checks, not a substitute for expert review.
- Visual grounding is simulated by task metadata in this first round.

## Next Steps

- Add real Hugging Face VLM wrappers.
- Add LoRA and QLoRA SFT scripts with PEFT and Accelerate.
- Add TRL DPO when the selected VLM prompt/image format is confirmed.
- Add OCR-enhanced plot understanding and figure panel detection.
- Add optional OpenAI-compatible judge evaluation.
- Expand to expert-curated materials VQA and caption datasets.
