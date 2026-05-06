import gc
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

LLM_DIR = Path(__file__).resolve().parents[1]
FINE_TUNING_DIR = LLM_DIR / "fine_tuning"

DEFAULT_MODEL_NAME = "google/gemma-4-E4B-it"
DEFAULT_SYSTEM_PROMPT = (
    "You are an NLU system for a human service robot. Given a user utterance, "
    "respond ONLY with a valid JSON object containing the detected intents and "
    "entities. Never respond with natural language or markdown. Output only JSON."
)

LORA_CONFIG = dict(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    use_rslora=False,
    task_type="CAUSAL_LM",
    ensure_weight_tying=True,
)

TRAIN_CONFIG = dict(
    max_length=512,
    num_train_epochs=4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    max_grad_norm=0.3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    seed=42,
    logging_steps=10,
    save_strategy="no",
    eval_strategy="steps",
    eval_steps=200,
    optim="paged_adamw_8bit",
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": True,
    },
)


def build_output_paths(model_name):
    model_dir = model_name.replace("/", "_")
    output = FINE_TUNING_DIR / "output" / model_dir
    output_merged = FINE_TUNING_DIR / "output_merged" / model_dir
    return output, output_merged


def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. Make sure CUDA drivers are installed.")

    name = torch.cuda.get_device_name()
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    log.info(f"GPU found: {name}")
    log.info(f"GPU mem: {mem:.1f} GB")

    return dict(dtype=torch_dtype, device_map="auto")


def load_causal_language_model(model_name, gpu):
    log.info(f"Loading model: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=gpu["dtype"],
        bnb_4bit_quant_storage=gpu["dtype"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=gpu["device_map"],
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    log.info(f"Parameters: {model.num_parameters() / 1e6:.2f} M")
    return model, tokenizer


def prepare_dataset(example):
    messages = example["messages"]

    if len(messages) >= 3 and messages[0]["role"] == "system":
        return {
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": messages[1]["content"]},
                {"role": "assistant", "content": messages[2]["content"]},
            ]
        }

    if len(messages) >= 2:
        return {
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": messages[0]["content"]},
                {"role": "assistant", "content": messages[1]["content"]},
            ]
        }

    raise ValueError("Each sample must contain at least user and assistant messages.")


def load_and_prepare_dataset(path):
    dataset = load_dataset("json", data_files=str(path), split="train")
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
    return dataset


def load_data(data=None, train_data=None, eval_data=None):
    if train_data and eval_data:
        log.info(f"Loading train data: {train_data}")
        log.info(f"Loading eval data: {eval_data}")
        train_set = load_and_prepare_dataset(train_data)
        test_set = load_and_prepare_dataset(eval_data)
    else:
        log.info(f"Preparing data: {data}")
        dataset = load_and_prepare_dataset(data)
        split = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
        train_set = split["train"]
        test_set = split["test"]

    log.info(f"Train samples: {len(train_set)}")
    log.info(f"Eval samples: {len(test_set)}")
    return train_set, test_set


def train(model, tokenizer, train_data, eval_data, output_dir):
    log.info("Starting training")

    train_config = dict(TRAIN_CONFIG)
    train_config["output_dir"] = output_dir

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=LoraConfig(**LORA_CONFIG),
        args=SFTConfig(
            fp16=True if model.dtype == torch.float16 else False,
            bf16=True if model.dtype == torch.bfloat16 else False,
            **train_config,
        ),
    )

    trainer.train()
    return trainer


def create_plot(trainer, output_dir):
    log_history = trainer.state.log_history

    train_losses = [entry["loss"] for entry in log_history if "loss" in entry]
    epoch_train = [entry["epoch"] for entry in log_history if "loss" in entry]
    eval_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]
    epoch_eval = [entry["epoch"] for entry in log_history if "eval_loss" in entry]

    plt.plot(epoch_train, train_losses, label="Training Loss")
    if eval_losses:
        plt.plot(epoch_eval, eval_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    path = os.path.join(output_dir, "train-loss.png")
    plt.savefig(path, dpi=300)
    plt.close()

    log.info(f"Train Loss Plot saved in {path}")


def save(trainer, tokenizer, output_dir):
    log.info(f"Saving LoRA & Tokenizer in: {output_dir}")

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    log.info("Done.")


def save_merged(model_name, output_dir, output_merged):
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    peft_model = PeftModel.from_pretrained(model, output_dir)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(output_merged, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    tokenizer.save_pretrained(output_merged)

    log.info("Merge complete.")
    log.info(f"Saved in {output_merged}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=Path, help="Path to one combined JSONL file")
    parser.add_argument("--train-data", type=Path, help="Path to the train split JSONL")
    parser.add_argument("--eval-data", type=Path, help="Path to the eval split JSONL")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name on Hugging Face",
    )
    args = parser.parse_args()

    if args.train_data and args.eval_data:
        data = None
    elif args.data:
        data = args.data
    else:
        parser.error("Provide either --data or both --train-data and --eval-data.")

    try:
        gpu = check_gpu()
        output_dir, output_merged = build_output_paths(args.model)
        model, tokenizer = load_causal_language_model(args.model, gpu)
        train_data, eval_data = load_data(data, args.train_data, args.eval_data)
        trainer = train(model, tokenizer, train_data, eval_data, output_dir)

        save(trainer, tokenizer, output_dir)
        create_plot(trainer, output_dir)

        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        save_merged(args.model, output_dir, output_merged)
    except KeyboardInterrupt:
        sys.exit(0)
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
