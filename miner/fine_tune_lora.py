#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class FineTuneConfig:
    model: str = "pearl-ai/Llama-3.3-70B-Instruct-pearl"
    dataset: str = ""
    dataset_split: str = "train"
    output_dir: str = "outputs/pearl-lora"
    text_field: str = "text"
    max_seq_length: int = 2048
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2
    warmup_ratio: float = 0.03
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    load_in_4bit: bool = False
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = True
    merge: bool = False
    trust_remote_code: bool = True


def parse_target_modules(value: str) -> tuple[str, ...]:
    modules = tuple(item.strip() for item in value.split(",") if item.strip())
    if not modules:
        raise ValueError("at least one LoRA target module is required")
    return modules


def extract_text(example: dict[str, object], tokenizer: object, text_field: str) -> str:
    if text_field in example and example[text_field] is not None:
        return str(example[text_field])
    if "prompt" in example and "completion" in example:
        return f"{example['prompt']}\n{example['completion']}"
    messages = example.get("messages")
    if isinstance(messages, list):
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            return str(
                apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return "\n".join(
            f"{message.get('role', 'unknown')}: {message.get('content', '')}"
            for message in messages
            if isinstance(message, dict)
        )
    raise ValueError(f"example must contain '{text_field}', prompt+completion, or messages fields")


def load_training_dataset(datasets_module: object, config: FineTuneConfig) -> object:
    load_dataset = datasets_module.load_dataset
    suffix = Path(config.dataset).suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return load_dataset("json", data_files=config.dataset, split=config.dataset_split)
    if suffix == ".csv":
        return load_dataset("csv", data_files=config.dataset, split=config.dataset_split)
    return load_dataset(config.dataset, split=config.dataset_split)


def build_training_arguments(
    config: FineTuneConfig,
    training_arguments_cls: Callable[..., object],
) -> object:
    return training_arguments_cls(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16,
        fp16=config.fp16,
        report_to="none",
        remove_unused_columns=False,
    )


def tokenize_dataset(dataset: object, tokenizer: object, config: FineTuneConfig) -> object:
    column_names = getattr(dataset, "column_names", None)

    def tokenize(example: dict[str, object]) -> dict[str, object]:
        text = extract_text(example, tokenizer, config.text_field)
        return tokenizer(
            text,
            max_length=config.max_seq_length,
            truncation=True,
            padding=False,
        )

    kwargs: dict[str, object] = {}
    if column_names:
        kwargs["remove_columns"] = column_names
    return dataset.map(tokenize, **kwargs)


def run_fine_tune(config: FineTuneConfig) -> None:
    import datasets  # noqa: PLC0415 - optional fine-tune dependency.
    from peft import (  # noqa: PLC0415 - optional fine-tune dependency.
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (  # noqa: PLC0415 - optional fine-tune dependency.
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, object] = {
        "device_map": "auto",
        "trust_remote_code": config.trust_remote_code,
    }
    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16" if config.bf16 else "float16",
        )

    model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    dataset = load_training_dataset(datasets, config)
    tokenized = tokenize_dataset(dataset, tokenizer, config)
    training_args = build_training_arguments(config, TrainingArguments)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    if config.merge:
        merged_dir = str(Path(config.output_dir) / "merged")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal LoRA SFT fine-tune entrypoint for Pearl-compatible vLLM models."
    )
    parser.add_argument("--model", default=FineTuneConfig.model)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--output-dir", default=FineTuneConfig.output_dir)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default=",".join(DEFAULT_TARGET_MODULES),
        help="comma-separated module names, for example q_proj,v_proj,o_proj",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> FineTuneConfig:
    return FineTuneConfig(
        model=args.model,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        output_dir=args.output_dir,
        text_field=args.text_field,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=parse_target_modules(args.lora_target_modules),
        load_in_4bit=args.load_in_4bit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        merge=args.merge,
        trust_remote_code=not args.no_trust_remote_code,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run_fine_tune(config_from_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
