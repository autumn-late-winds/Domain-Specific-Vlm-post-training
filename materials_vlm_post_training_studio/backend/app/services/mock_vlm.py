DOMAIN_TERMS = {
    "sem_description": "porous nanosheet-like morphology with hierarchical aggregation",
    "caption_generation": "scientific figure showing materials characterization features",
    "chart_understanding": "Sample B has the strongest plotted response",
    "materials_qa": "the image evidence supports a materials-specific answer",
    "spectrum_interpretation": "the spectrum contains characteristic peaks linked to phase or bonding information",
}


class MockVLM:
    def __init__(self, variant: str = "base") -> None:
        self.variant = variant

    def generate(self, question: str, task_type: str, metadata: dict | None = None) -> str:
        modality = (metadata or {}).get("modality", "materials image")
        if self.variant == "base":
            return f"This appears to be a {modality}. The figure shows materials-related features."
        detail = DOMAIN_TERMS.get(task_type, DOMAIN_TERMS["materials_qa"])
        return (
            f"For this {modality} task, the answer is grounded in the visual pattern: {detail}. "
            f"I would avoid inferring composition or synthesis conditions unless they are explicitly shown."
        )

