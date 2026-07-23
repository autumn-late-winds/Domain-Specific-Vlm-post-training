import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.dataset_service import dataset_stats, validate_dataset
from app.services.evaluation_service import evaluate_experiment, keyword_coverage, token_f1
from app.services.report_service import report_markdown
from app.services.training_service import start_mock_training


def test_dataset_validation_passes():
    result = validate_dataset("demo_materials_vlm")
    assert result["valid"] is True
    assert result["errors"] == []


def test_dataset_stats_include_tasks():
    stats = dataset_stats("demo_materials_vlm")
    assert stats["total_examples"] >= 3
    assert "sem_description" in stats["by_task_type"]


def test_metrics_reward_domain_terms():
    assert token_f1("porous nanosheet morphology", "porous nanosheet morphology") == 1.0
    assert keyword_coverage("porous nanosheet hierarchical morphology") > 0


def test_mock_training_and_report_generation():
    summary = start_mock_training("demo_materials_vlm", "mock-vlm-materials", epochs=1)
    report = evaluate_experiment(summary["experiment_id"], "demo_materials_vlm")
    markdown = report_markdown(summary["experiment_id"])
    assert summary["status"] == "completed"
    assert report["aggregate_metrics"]["delta_overall"] >= 0
    assert "Materials VLM Post Training Report" in markdown

