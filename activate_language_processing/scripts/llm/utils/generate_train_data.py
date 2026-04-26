import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from ollama import chat
from jsonschema import validate, ValidationError
from tqdm import tqdm

INTENTS = [
    "guide",
    "navigation",
    "pick_up",
    "place",
    "affirm",
    "deny",
    "lookup",
    # "clarify",
    "talk_with_human",
    "follow",
    "count",
]
ROLES = ["Person", "SourceRoom", "DestinationRoom", "Furniture", "Clothes", "Item"]
ENTITY = ["NaturalPerson", "Room", "DesignedFurniture", "Clothing", "Transportable"]

# JSON schema that we want to keep
SCHEMA = {
    "type": "object",
    "required": ["intents"],
    "properties": {
        "intents": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["intent", "entities"],
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": INTENTS,
                    },
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "role",
                                "value",
                                "entity",
                                "propertyAttribute",
                                "actionsAttribute",
                                "numberAttribute",
                            ],
                            "properties": {
                                "role": {"type": "string", "enum": ROLES},
                                "value": {"type": "string"},
                                "entity": {"type": "string", "enum": ENTITY},
                                "propertyAttribute": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "actionsAttribute": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "numberAttribute": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
            "minItems": 1,
        }
    },
}

EXAMPLE = json.dumps(
    {
        "intents": [
            {
                "intent": "pick_up",
                "entities": [
                    {
                        "role": "Item",
                        "value": "ice tea",
                        "entity": "Transportable",
                        "propertyAttribute": [],
                        "actionsAttribute": [],
                        "numberAttribute": [],
                    },
                    {
                        "role": "Furniture",
                        "value": "arm chair",
                        "entity": "DesignedFurniture",
                        "propertyAttribute": [],
                        "actionsAttribute": [],
                        "numberAttribute": [],
                    },
                ],
            },
            {
                "intent": "place",
                "entities": [
                    {
                        "role": "Item",
                        "value": "ice tea",
                        "entity": "Transportable",
                        "propertyAttribute": [],
                        "actionsAttribute": [],
                        "numberAttribute": [],
                    },
                    {
                        "role": "Furniture",
                        "value": "kitchen counter",
                        "entity": "DesignedFurniture",
                        "propertyAttribute": [],
                        "actionsAttribute": [],
                        "numberAttribute": [],
                    },
                ],
            },
        ]
    },
    indent=2,
)


# Load generated sentences from the official RoboCup@Home command generator
def load_sentences(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def sentences_to_json(sentence, model):
    PROMPT = f"""
Convert the following instruction into structured JSON.

Instruction:
{sentence}

Return ONLY JSON in the following schema:
{json.dumps(SCHEMA)}

Example: 
Instruction = 'Get an ice tea from the arm chair and place it on the kitchen counter'
Result = {EXAMPLE}

Rules:
- ONLY valid intents: {INTENTS}
- ONLY valid entities: {ENTITY}
- ONLY valid roles: {ROLES}
- numberAttributes must be a list of WORDS ("two" instead of 2)
- actionAttributes contains a list of actions like "waving", "pointing", "sitting"...
- propertyAttributes must be a list of attributes like colors
- "intents" must be a list of objects
- each intent must have:
    - "intent": one of {INTENTS}
    - "entities": list of entities ONLY relevant to that intent
- do NOT mix entities between intents
- NO markdown
- NO explanation
"""

    response = chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        think=False,
        stream=False,
        format="json",
        options={"temperature": 0.7},
    )
    return json.loads(response.message.content.strip())


def json_validation(data):
    try:
        validate(instance=data, schema=SCHEMA)
        return True, None
    except ValidationError as e:
        return False, e


def paraphrase_sentence(sentence, model, n=3):
    PROMPT = f"""
You are tasked with generating **{n}** task commands.

Your input will be a **single task command**.
Your output must be **a single command** containing **the alternative phrasing** of that command **and nothing else**.

Given command:
{sentence}

---

### **Guidelines**

* **Complexity gradient:**

  * The **first paraphrase** should use the **most complex or formal** sentence structure.
  * Each subsequent paraphrase should become **progressively simpler and more natural**.

* **Content preservation:**

  * Keep all **entities, objects, and locations exactly the same** (e.g., "coke" must remain "coke").
  * You may **restructure the sentence** as long as meaning and entities are preserved.

* **Tone and style:**

  * Maintain a **natural, conversational tone** write as if real people might say it.
  * Avoid robotic or overly formal phrasing unless required for the most complex version.

Return ONLY valid JSON, e.g.: {{"variants": ["phrasing one", "phrasing two", "phrasing three"]}}
NO markdown. NO explanation.
"""

    response = chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        think=False,
        stream=False,
        format="json",
        options={"temperature": 0.9},
    )

    response = json.loads(response.message.content.strip())
    return [str(v) for v in response.get("variants", [])]


def post_process_samples(sample):
    system_prompt = (
        "You are an NLU system for a human service robot. Given a user utterance, "
        "respond ONLY with a valid JSON object containing the detected intents and entities. "
        "Never respond with natural language or markdown. Output only JSON."
    )

    seen = set()
    results = []
    for sentence in sample["variants"]:
        if sentence not in seen:
            seen.add(sentence)
            results.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": sentence},
                        {"role": "assistant", "content": json.dumps(sample["label"])},
                    ]
                }
            )
    return results


def check_correctness_of_data(model, sentence, label):
    prompt = f"""
Does the JSON correctly represent the instruction?

Instruction:
{sentence}

JSON:
{json.dumps(label)}

Correct schema:
{json.dumps(SCHEMA)}

Rules to check:
- All intents match the actions described in the instruction.
- All entities (objects, rooms, furniture, people) from the instruction are present in the JSON.
- Roles and entity types are appropriate for the values.
- No extra intents or entities are fabricated that aren't in the instruction.
- All attributes are appropriate for the values (color, actions, numbers).
- Numbers in 'numberAttribute' are always written numbers (one, two, three, etc.). Leave it EMPTY [] if no explicit number is stated (e.g. "how many X" has no numberAttribute - it queries a count, it does not state one)

Answer only YES or NO.
If NO, explain why the sample is not correct. YES or NO should be in the very first position in your answer. 
"""

    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        think=False,
        stream=False,
        options={"temperature": 0.0},
    )

    raw = response.message.content.strip()
    return raw.upper().startswith("YES"), raw


def process_sentences(model, sentence, num_variants):
    json_label = sentences_to_json(sentence, model)
    ok, err = json_validation(json_label)
    if not ok:
        tqdm.write(f"Invalid sample skipped: {err}")
        return None

    variants = [sentence]
    try:
        variants.extend(paraphrase_sentence(sentence, model, n=num_variants))
    except Exception:
        # proceed with original
        pass

    return {"sentence": sentence, "variants": variants, "label": json_label}


def collect_dataset(sentences, model, num_variants, workers):
    dataset = []
    lock = Lock()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_sentences, model, s, num_variants): s
            for s in sentences
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing sentences"
        ):
            s = futures[future]
            try:
                entry = future.result()
                if entry is not None:
                    with lock:
                        dataset.append(entry)
            except Exception as e:
                tqdm.write(f"Error processing '{s}': {e}")

    return dataset


def verify_dataset(dataset, check_model, workers):
    verified, rejected = [], []
    lock = Lock()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                check_correctness_of_data, check_model, s["sentence"], s["label"]
            ): s
            for s in dataset
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Verifying data"
        ):
            sample = futures[future]
            try:
                is_correct, explanation = future.result()
            except Exception as e:
                tqdm.write(f"Verification error for '{sample['sentence']}': {e}")
                is_correct, explanation = False, ""  # reject on error

            with lock:
                if is_correct:
                    verified.append(sample)
                else:
                    tqdm.write(f"Rejected: {sample['sentence']}")
                    rejected.append(
                        {
                            "sentence": sample["sentence"],
                            "label": sample["label"],
                            "reason": explanation,
                        }
                    )

    return verified, rejected


def save_dataset(dataset, save_path, tuning_path):
    with open(save_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {len(dataset)} samples to {save_path}")

    chatml_samples = []
    for sample in tqdm(dataset, desc="Converting to ChatML"):
        chatml_samples.extend(post_process_samples(sample))

    with open(tuning_path, "w") as f:
        for s in chatml_samples:
            f.write(json.dumps(s) + "\n")
    print(f"Saved {len(chatml_samples)} ChatML samples to {tuning_path}")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model used for labelling and paraphrasing.",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        required=True,
        help="Path to the file with one sentence per line.",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=Path,
        default="../fine_tuning/llm_results.json",
        help="Where to save the raw labelled dataset (.json).",
    )
    parser.add_argument(
        "-t",
        "--tuning_data",
        type=Path,
        default="../fine_tuning/llm_training_data.jsonl",
        help="Where to save the ChatML fine-tune dataset (.jsonl).",
    )
    parser.add_argument(
        "-check",
        "--check_data",
        action="store_true",
        default=False,
        help="Verify each sample with an LLM and remove likely incorrect ones.",
    )
    parser.add_argument(
        "--check_model",
        type=str,
        default=None,
        help=(
            "Model used for correctness verification. "
            "Can be a smaller/faster model than --model. "
            "Defaults to --model when not set."
        ),
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker threads (default: 4).",
    )
    parser.add_argument(
        "--check_workers",
        type=int,
        default=None,
        help=(
            "Number of parallel workers for verification. "
            "Defaults to --workers when not set. "
            "Useful when --check_model is a smaller model that can handle more concurrency."
        ),
    )
    parser.add_argument(
        "-n",
        "--num_variants",
        type=int,
        default=3,
        help="Number of paraphrase variants per sentence (default: 3).",
    )
    return parser.parse_args()


def main():
    args = args_parser()

    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.tuning_data.parent.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences(args.data)
    dataset = []

    try:
        dataset = collect_dataset(
            sentences, args.model, args.num_variants, args.workers
        )

        if args.check_data:
            check_model = args.check_model or args.model
            check_workers = args.check_workers or args.workers
            dataset, rejected = verify_dataset(dataset, check_model, check_workers)
            print(
                f"Verification complete. Kept {len(dataset)}, removed {len(rejected)}"
            )

            report_path = args.save.with_stem(args.save.stem + "_rejected")
            with open(report_path, "w") as f:
                json.dump(rejected, f, indent=2)
            print(f"Saved rejected samples to {report_path}")

    except KeyboardInterrupt:
        print("\nSaving data...")

    finally:
        save_dataset(dataset, args.save, args.tuning_data)


if __name__ == "__main__":
    main()
