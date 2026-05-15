"""
LoRA Training Script — Fine-tune Gemma 4 with QLoRA.

Uses trl.SFTTrainer for supervised fine-tuning with PEFT adapters.
Run: python -m src.fine_tuning.train
"""

import logging
from pathlib import Path

from .lora_config import load_ft_config, get_lora_config, get_quantization_config

logger = logging.getLogger("fine_tuning.train")


def train():
    """Run LoRA fine-tuning on Gemma 4."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import load_dataset
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Install: pip install peft trl datasets bitsandbytes")
        return

    config = load_ft_config()
    base_cfg = config["base_model"]
    training_cfg = config["training"]
    dataset_cfg = config["dataset"]

    model_name = base_cfg["name"]
    output_dir = training_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base model: {model_name}")

    # Load quantized model
    quant_config = get_quantization_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("json", data_files=dataset_cfg["train_file"], split="train")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=training_cfg.get("learning_rate", 2e-4),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        warmup_steps=training_cfg.get("warmup_steps", 100),
        logging_steps=training_cfg.get("logging_steps", 10),
        save_steps=training_cfg.get("save_steps", 200),
        bf16=training_cfg.get("bf16", True),
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        max_seq_length=training_cfg.get("max_seq_length", 2048),
    )

    logger.info("Starting LoRA fine-tuning...")
    trainer.train()

    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"LoRA adapter saved to: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
