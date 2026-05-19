"""
Adapter Loader — Load LoRA adapter weights at inference time.
"""

import logging
from pathlib import Path
from typing import Optional

from .lora_config import load_ft_config

logger = logging.getLogger("fine_tuning.adapter_loader")


class AdapterLoader:
    """Loads and manages LoRA adapter for Gemma 4."""

    def __init__(self):
        self.config = load_ft_config()
        self.adapter_path = self.config.get("adapter", {}).get("path", "data/fine_tuned/lora_adapter")
        self.auto_load = self.config.get("adapter", {}).get("auto_load", False)
        self._model = None
        self._tokenizer = None

    @property
    def is_available(self) -> bool:
        """Check if a trained adapter exists."""
        path = Path(self.adapter_path)
        return path.exists() and (path / "adapter_config.json").exists()

    def load(self):
        """Load the LoRA adapter onto the base model."""
        if not self.is_available:
            logger.warning(f"No adapter found at {self.adapter_path}. Run training first.")
            return None

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            base_model_name = self.config["base_model"]["name"]
            logger.info(f"Loading base model: {base_model_name}")

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
            )

            logger.info(f"Loading LoRA adapter from: {self.adapter_path}")
            self._model = PeftModel.from_pretrained(base_model, self.adapter_path)
            self._tokenizer = AutoTokenizer.from_pretrained(base_model_name)

            logger.info("LoRA adapter loaded successfully")
            return self._model

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return None

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate text using the fine-tuned model."""
        if self._model is None:
            self.load()
        if self._model is None:
            return ""

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
