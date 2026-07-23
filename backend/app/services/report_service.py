from app.services.evaluation_service import get_report


def report_markdown(experiment_id: str) -> str:
    report = get_report(experiment_id)
    metrics = report["aggregate_metrics"]
    lines = [
        f"# Materials VLM Post Training Report: {experiment_id}",
        "",
        f"Mode: `{report['mode']}`",
        "",
        "## Summary",
        report["qualitative_summary"],
        "",
        "## Aggregate Metrics",
        f"- Base overall: {metrics['base']['overall']}",
        f"- Fine-tuned overall: {metrics['finetuned']['overall']}",
        f"- Delta overall: {metrics['delta_overall']}",
        "",
        "## Representative Examples",
    ]
    for row in report["before_after_examples"][:3]:
        lines.extend(
            [
                f"### {row['id']} ({row['task_type']})",
                f"Question: {row['question']}",
                f"Base: {row['base_prediction']}",
                f"Fine-tuned: {row['finetuned_prediction']}",
                "",
            ]
        )
    lines.extend(["## Limitations", *[f"- {item}" for item in report["limitations"]]])
    return "\n".join(lines)

