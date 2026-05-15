"""
LoRA Config — Parameter-efficient fine-tuning setup for Gemma 4.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("fine_tuning.lora_config")

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "fine_tuning_config.yaml"


def load_ft_config(path: Optional[str] = None) -> dict:
    """Load fine-tuning configuration from YAML."""
    config_path = Path(path) if path else _CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_lora_config():
    """Create a PEFT LoraConfig from YAML settings."""
    try:
        from peft import LoraConfig, TaskType

        config = load_ft_config()
        lora_cfg = config.get("lora", {})

        return LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
    except ImportError:
        logger.warning("peft not installed. Install with: pip install peft")
        return None


def get_quantization_config():
    """Create BitsAndBytes quantization config for QLoRA."""
    try:
        from transformers import BitsAndBytesConfig
        import torch

        config = load_ft_config()
        base_cfg = config.get("base_model", {})
        quant = base_cfg.get("quantization", "4bit")

        if quant == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif quant == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None
    except ImportError:
        logger.warning("bitsandbytes not installed.")
        return None
