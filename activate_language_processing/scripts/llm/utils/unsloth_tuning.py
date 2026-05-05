import os
import argparse
import torch
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset
from unsloth import (
    FastModel,
    get_chat_template,
    train_on_responses_only,
)
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq, TextStreamer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_CONFIGS = {
    # vision_processor=True  → content must be [{"type":"text","text":"..."}]
    # vision_processor=False → content must be a plain string
    "llama-3": {
        "template": "llama-3.1",
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "vision_processor": False,
    },
    "ministral": {
        "template": "mistral",
        "instruction_part": "[INST] ",
        "response_part": "[/INST]",
        "vision_processor": True,
    },
    "mistral": {
        "template": "mistral",
        "instruction_part": "[INST] ",
        "response_part": "[/INST]",
        "vision_processor": False,
    },
    "qwen": {
        "template": "qwen25",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        # Qwen3.5 is multimodal but its Jinja template expects plain strings
        "vision_processor": False,
    },
    "gemma": {
        "template": "gemma3",
        "instruction_part": "<start_of_turn>user\n",
        "response_part": "<start_of_turn>model\n",
        "vision_processor": False,
    },
}


def get_model_config(model_name):
    name_lower = model_name.lower()
    for key, config in MODEL_CONFIGS.items():
        if key in name_lower:
            return config
    log.warning(
        f"Unknown model architecture for {model_name}. Defaulting to llama-3 template."
    )
    return MODEL_CONFIGS["llama-3"]


def build_output_paths(model_name, base_dir):
    safe_name = model_name.rsplit("/", 1)[-1].replace(".", "_")
    model_dir = safe_name
    output = os.path.join(base_dir, model_dir, "lora")
    output_gguf = os.path.join(base_dir, model_dir, "gguf")
    return output, output_gguf


def load_and_prepare_dataset(file_path, tokenizer):
    log.info(f"Loading dataset from {file_path}")
    dataset = load_dataset("json", data_files=str(file_path), split="train")

    def formatting_prompts_func(examples):
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in examples["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset


def plot_training_loss(log_history, save_path):
    """Extract training loss from trainer log_history and save a line plot."""
    train_losses = [entry["loss"] for entry in log_history if "loss" in entry]
    epoch_train = [entry["epoch"] for entry in log_history if "loss" in entry]
    eval_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]
    epoch_eval = [entry["epoch"] for entry in log_history if "eval_loss" in entry]

    fig, ax = plt.subplots()
    ax.plot(epoch_train, train_losses, label="Training Loss")
    if eval_losses:
        ax.plot(epoch_eval, eval_losses, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train-Loss Validation")
    ax.legend()
    ax.grid(True)

    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    log.info(f"Train Loss Plot saved in {save_path}")


def run_inference_tests(model, tokenizer, config):
    """Run a few inference tests after training."""
    tokenizer = get_chat_template(tokenizer, chat_template=config["template"])
    FastModel.for_inference(model)

    def make_content(text):
        # Ministral use a vision processor that requires
        # structured content dicts. All other models expect a plain string.
        if config.get("vision_processor", False):
            return [{"type": "text", "text": text}]
        return text

    def generate(messages, max_new_tokens=2048, stream=False):
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=1.0,
            min_p=0.1,
        )
        if stream:
            kwargs["streamer"] = TextStreamer(tokenizer, skip_prompt=True)
        return model.generate(**kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Test 1: Robot command interpretation (main use case) ---
    print("\n" + "=" * 70)
    print("INFERENCE TEST - Robot command")
    print("=" * 70)

    system_prompt = (
        "You are an NLU system for a human service robot. "
        "Given a user utterance, respond ONLY with a valid JSON object "
        "containing the detected intents and entities. "
        "Never respond with natural language or markdown. Output only JSON."
    )

    messages = [
        {"role": "system", "content": make_content(system_prompt)},
        {
            "role": "user",
            "content": make_content(
                "Please bring me a cola from the kitchen and place it on the table."
            ),
        },
    ]

    outputs = generate(messages, max_new_tokens=512)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print("Command output:")
    print(response[0])

    # --- Test 2: General knowledge (Fibonacci) -----------------
    print("\n" + "=" * 70)
    print("INFERENCE TEST - General knowledge (Fibonacci)")
    print("=" * 70)

    messages = [
        {
            "role": "user",
            "content": make_content(
                "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,"
            ),
        },
    ]
    generate(messages, max_new_tokens=128, stream=True)
    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Unsloth Fine-tuning Script")
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default="../fine_tuning/llm_training_data.jsonl",
        help="Path to JSONL training data",
    )
    parser.add_argument(
        "--epochs", type=int, default=4, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size per device"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Max sequence length"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../fine_tuning/unsloth_output",
        help="Base directory for all models",
    )

    args = parser.parse_args()

    # 1. Path setup
    lora_path, gguf_path = build_output_paths(args.model, args.output_dir)
    loss_plot_path = os.path.join(lora_path, "training_loss.png")

    # 2. Load Model & Tokenizer
    log.info(f"Loading model: {args.model}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        device_map="auto",
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # skip vision encoder - text-only fine-tuning
        finetune_language_layers=True,
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        random_state=42,
    )

    config = get_model_config(args.model)
    tokenizer = get_chat_template(tokenizer, chat_template=config["template"])

    if not args.data.exists():
        log.error(f"Data file not found: {args.data}")
        return

    dataset = load_and_prepare_dataset(args.data, tokenizer)

    # 3. Trainer Configuration (checkpoint saving disabled)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            warmup_ratio=0.03,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=lora_path,
            report_to="none",
            # Disable all checkpoint saving
            save_strategy="no",
            save_steps=0,
            save_total_limit=0,
        ),
    )

    # Train only on responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part=config["instruction_part"],
        response_part=config["response_part"],
    )

    log.info("Starting training process...")
    trainer.train()

    # 4. Generate loss plot
    plot_training_loss(trainer.state.log_history, save_path=loss_plot_path)

    # 5. Save final adapter and optionally GGUF
    log.info(f"Saving LoRA adapter to {lora_path}")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)

    log.info(f"Exporting to GGUF format in {gguf_path}")
    try:
        import shutil, tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_gguf_src = tmp_dir + "_gguf"
            model.save_pretrained_gguf(tmp_dir, tokenizer, quantization_method="q4_k_m")
            os.makedirs(gguf_path, exist_ok=True)
            if os.path.isdir(tmp_gguf_src):
                modelfile_created = False
                gguf_filename = None
                for fname in os.listdir(tmp_gguf_src):
                    # Skip vision projection weights — text-only inference
                    if "mmproj" in fname.lower():
                        log.info(f"Skipping vision projection file: {fname}")
                        continue
                    shutil.move(
                        os.path.join(tmp_gguf_src, fname),
                        os.path.join(gguf_path, fname),
                    )
                    if fname.endswith(".gguf"):
                        gguf_filename = fname
                    if fname == "Modelfile":
                        modelfile_created = True

                # unsloth skips the Modelfile for some models (e.g. Qwen3.5)
                # generate a minimal one so Ollama can load the model directly
                if not modelfile_created and gguf_filename:
                    modelfile_path = os.path.join(gguf_path, "Modelfile")
                    with open(modelfile_path, "w") as mf:
                        mf.write(f"FROM ./{gguf_filename}\n")
                    log.info(f"Generated fallback Modelfile -> {modelfile_path}")

                shutil.rmtree(tmp_gguf_src, ignore_errors=True)
            else:
                log.warning(f"Expected GGUF dir not found: {tmp_gguf_src}")
        log.info(f"GGUF saved to {gguf_path}")

    except Exception as e:
        # Fallback: save merged safetensors and convert manually with:
        #   python llama.cpp/convert_hf_to_gguf.py <gguf_path> --outtype q4_k_m
        log.warning(
            f"GGUF export failed ({e}). "
            f"Falling back to merged safetensors in {gguf_path}"
        )
        try:
            import shutil

            model.save_pretrained_merged(
                gguf_path, tokenizer, save_method="merged_16bit"
            )
            log.info(f"Merged safetensors saved to {gguf_path}")
        except Exception as e2:
            log.error(f"Merged safetensors export also failed: {e2}")

    log.info("Process finished successfully.")

    # 6. Run inference tests
    run_inference_tests(model, tokenizer, config)


if __name__ == "__main__":
    main()
