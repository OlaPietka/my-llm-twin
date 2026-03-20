"""
LoRA fine-tuning via PEFT and TRL.

Takes a JSONL dataset of chat-formatted examples and fine-tunes
a base model with LoRA adapters.
"""

import json
import logging
from pathlib import Path

from my_llm_twin.config import TrainingConfig, DatasetConfig

logger = logging.getLogger("my_llm_twin")


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def run_training(
    training_config: TrainingConfig,
    dataset_config: DatasetConfig,
    dataset_dir: Path,
):
    # heavy imports inside the function so the rest of the CLI stays fast
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for training. "
            "Make sure you have an NVIDIA GPU with CUDA installed."
        )

    train_path = dataset_dir / "train.jsonl"
    val_path = dataset_dir / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. Run 'build-dataset' first."
        )

    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path) if val_path.exists() else []

    logger.info(f"Training examples: {len(train_data)}")
    logger.info(f"Validation examples: {len(val_data)}")

    if len(train_data) < 100:
        logger.warning(
            f"Only {len(train_data)} training examples. "
            "Results may be poor — consider adding more conversation data."
        )

    # load model and tokenizer
    logger.info(f"Loading model: {training_config.base_model}")
    dtype = torch.bfloat16 if training_config.precision == "bf16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(training_config.base_model)

    # register <|msg|> as a special token so it doesn't get split
    special_tokens = {"additional_special_tokens": [dataset_config.separator]}
    tokenizer.add_special_tokens(special_tokens)

    # Load in 4-bit to fit in 16GB VRAM with room for LoRA + optimizer states
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        training_config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    lora_config = LoraConfig(
        r=training_config.lora_rank,
        lora_alpha=training_config.lora_alpha,
        target_modules=training_config.lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # prepare datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None

    output_dir = Path(training_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=training_config.epochs,
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        max_length=training_config.max_seq_length,
        logging_steps=10,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=50 if val_dataset else None,
        save_strategy="epoch",
        bf16=training_config.precision == "bf16",
        fp16=training_config.precision == "fp16",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    # save adapter and tokenizer
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Model saved to {output_dir}")
