from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .vlm_wrapper import VLMWrapper


@dataclass
class Qwen3VLWrapperConfig:
    model_name_or_path: str
    device_map: str = "auto"
    torch_dtype: str = "auto"
    use_4bit: bool = False
    trust_remote_code: bool = True
    local_files_only: bool = True
    attn_implementation: str | None = None


class Qwen3VLWrapper(VLMWrapper):
    """Qwen3-VL adapter that never downloads weights unless explicitly configured.

    The wrapper keeps imports lazy so mock mode and unit tests do not require
    heavy training dependencies. It expects model files to already exist in the
    local Hugging Face cache or at a local path when local_files_only=True.
    """

    def __init__(self, config: Qwen3VLWrapperConfig) -> None:
        self.config = config
        self.processor: Any | None = None
        self.model: Any | None = None

    def load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-VL real mode requires optional dependencies: "
                "torch, transformers, accelerate, peft, pillow, qwen-vl-utils, and optionally bitsandbytes."
            ) from exc

        quantization_config = None
        if self.config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        model_kwargs: dict[str, Any] = {
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
            "quantization_config": quantization_config,
        }
        if self.config.torch_dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype)
        else:
            model_kwargs["torch_dtype"] = "auto"
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=self.config.local_files_only,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.config.model_name_or_path,
                **model_kwargs,
            )
        except Exception as exc:
            download_hint = (
                "Model loading failed. This project defaults to local_files_only=True, so no model files were "
                "downloaded. Put Qwen3-VL files in the local HF cache or pass a local model path."
            )
            raise RuntimeError(download_hint) from exc

    def build_messages(self, image_path: str | Path, question: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": question},
                ],
            }
        ]

    def preprocess_for_training(self, example: dict[str, Any], image_root: str | Path | None = None) -> dict[str, Any]:
        image_path = Path(example["image"])
        if image_root is not None and not image_path.is_absolute():
            image_path = Path(image_root) / image_path.name
        messages = self.build_messages(image_path, example["question"])
        messages.append({"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]})
        return {"id": example.get("id"), "messages": messages, "task_type": example.get("task_type")}

    def generate(self, image_path: str | Path, question: str, max_new_tokens: int = 256) -> str:
        if self.model is None or self.processor is None:
            self.load_model()
        assert self.model is not None
        assert self.processor is not None

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError("Qwen3-VL inference requires qwen-vl-utils for image preprocessing.") from exc

        messages = self.build_messages(image_path, question)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def save_adapter(self, output_dir: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save adapter before loading/training a model.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)

