import argparse
import json
import os
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
        messages=[
            {
                "role": "user",
                "content": PROMPT,
            },
        ],
        think=False,
        stream=False,
        format="json",
        options={"temperature": 0.7},
    )
    response = response.message.content.strip()
    return json.loads(response)


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

  * Keep all **entities, objects, and locations exactly the same** (e.g., “coke” must remain “coke”).
  * You may **restructure the sentence** as long as meaning and entities are preserved.

* **Tone and style:**

  * Maintain a **natural, conversational tone** write as if real people might say it.
  * Avoid robotic or overly formal phrasing unless required for the most complex version.
  
Return ONLY valid JSON, e.g.: {{"variants": ["phrasing one", "phrasing two", "phrasing three"]}}
NO markdown. NO explanation.
"""

    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": PROMPT,
            },
        ],
        # reduce time
        think=False,
        stream=False,
        format="json",
        options={"temperature": 0.9},
    )

    response = json.loads(response.message.content.strip())
    return [str(v) for v in response.get("variants", [])]


def post_process_samples(sample):
    system_prompt = """You are an NLU system for a human service robot. Given a user utterance, respond ONLY with a valid JSON object containing the detected intents and entities. Never respond with natural language or markdown. Output only JSON."""

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
        messages=[
            {"role": "user", "content": prompt},
        ],
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


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The model to use"),
    parser.add_argument(
        "-d", "--data", type=Path, required=True, help="List of the sentences"
    ),
    parser.add_argument(
        "-s",
        "--save",
        type=Path,
        required=False,
        default="../fine_tuning/llm_results.json",
        help="Where to save the results (.json)",
    ),
    parser.add_argument(
        "-t",
        "--tuning_data",
        type=Path,
        help="Where to save the fine-tune data (.jsonl)",
        default="../fine_tuning/llm_training_data.jsonl",
    ),
    parser.add_argument(
        "-check",
        "--check_data",
        help="Whether to check the data with an LLM afterwards. This will remove potentially bad data from the dataset and the correctness is NOT guaranteed.",
        action="store_true",
        default=False,
    ),
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads (Default: 4)",
    ),
    parser.add_argument(
        "-n",
        "--num_variants",
        type=int,
        default=3,
        help="Number of sentence variants to use (Default: 3)",
    ),

    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    sentences = load_sentences(args.data)

    dataset = []
    lock = Lock()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_sentences, args.model, s, args.num_variants): s
                for s in sentences
            }
            for future in tqdm(
                as_completed(futures), total=len(sentences), desc="Processing sentences"
            ):
                s = futures[future]
                try:
                    entry = future.result()
                    if entry is not None:
                        with lock:
                            dataset.append(entry)
                except Exception as e:
                    tqdm.write(f"Error processing sentence: {s} {e}")

            # Check correctness of data with LLM if flagged
            if args.check_data:
                verified = []
                rejected = []
                for sample in tqdm(dataset, desc="Verifying data"):
                    is_correct, explanation = check_correctness_of_data(
                        args.model, sample["sentence"], sample["label"]
                    )
                    if is_correct:
                        verified.append(sample)
                    else:
                        tqdm.write(f"Incorrect sample found: {sample['sentence']}")
                        rejected.append(
                            {
                                "sentence": sample["sentence"],
                                "label": sample["label"],
                                "reason": explanation,
                            }
                        )
                removed = len(dataset) - len(verified)
                print(f"Removed {removed} out of {len(dataset)}")
                dataset = verified

                report_path = args.save.with_stem(args.save.stem + "_rejected")
                with open(report_path, "w") as f:
                    json.dump(rejected, f, indent=2)
                print(f"Saved rejected samples to {report_path}.")
    except KeyboardInterrupt:
        print("Saving data...")
    finally:
        # Save raw dataset
        with open(args.save, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved dataset with {len(dataset)} samples to {args.save}")

    # Convert to chatML and save
    final_dataset = []
    for sample in tqdm(dataset, desc="Processing samples"):
        final_dataset.extend(post_process_samples(sample))

    output_dir = os.path.dirname(args.tuning_data)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.tuning_data, "w") as f:
        # save train_data as JSONL.
        for s in final_dataset:
            f.write(json.dumps(s) + "\n")
    print(
        f"Saved chatML dataset with {len(final_dataset)} samples to {Path(args.tuning_data)}"
    )


if __name__ == "__main__":
    main()
