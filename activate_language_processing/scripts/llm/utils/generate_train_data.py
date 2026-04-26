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
    """Load command sentences from a text file.

    Empty lines are ignored. Each remaining line is stripped so the returned
    sentences can be passed directly into the data generation pipeline.

    The sentences were generated with the `RoboCup@Home` command generator.
    https://github.com/RoboCupAtHome/CommandGenerator

    Args:
        path: Path to a text file that contains one sentence per line.

    Returns:
        A list of cleaned, non-empty sentences.
    """
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def sentences_to_json(sentence, model):
    """Convert a single sentence into the target JSON label format.

    The function sends the sentence and the expected schema to the LLM and
    parses the model response as JSON. The returned object is expected to
    follow the intent and entity structure defined in ``SCHEMA``.

    Args:
        sentence: Natural-language robot command to label.
        model: Name of the Ollama model used for labeling.

    Returns:
        A dictionary containing the structured intents and entities.

    Raises:
        json.JSONDecodeError: If the model response is not valid JSON.
    """
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
    """Validate a generated label against the expected schema.

    Args:
        data: Parsed JSON label produced by the LLM.

    Returns:
        A tuple ``(is_valid, error)``. ``is_valid`` is ``True`` when the label
        matches ``SCHEMA``. ``error`` is ``None`` on success and the validation
        exception on failure.
    """
    try:
        validate(instance=data, schema=SCHEMA)
        return True, None
    except ValidationError as e:
        return False, e


def paraphrase_sentence(sentence, model, n=3):
    """Generate alternative phrasings for a sentence.

    The created variants should preserve the original meaning and entities
    while giving the dataset more language variety for training.

    Args:
        sentence: Original command sentence.
        model: Name of the Ollama model used for paraphrasing.
        n: Number of paraphrase variants to request from the model.

    Returns:
        A list of paraphrased sentence strings. If the response does not
        contain the expected key, an empty list is returned.

    Raises:
        json.JSONDecodeError: If the model response is not valid JSON.
    """
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
    """Convert one labeled sample into ChatML training examples.

    The function creates one ChatML record for each unique sentence variant.
    Every record contains a system prompt, the user sentence, and the target
    JSON label as the assistant response.

    Args:
        sample: Dataset entry with ``variants`` and ``label`` fields.

    Returns:
        A list of ChatML-formatted training samples.
    """
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
    """Check whether a label correctly matches a sentence.

    A second LLM is asked to review the generated label against the sentence
    and the schema rules. The model must answer with ``YES`` or ``NO`` and may
    include an explanation when the sample is rejected.

    Args:
        model: Name of the Ollama model used for verification.
        sentence: Original command sentence.
        label: Structured JSON label assigned to the sentence.

    Returns:
        A tuple ``(is_correct, raw_response)`` where ``is_correct`` is based on
        whether the model answer starts with ``YES``.
    """
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
    """Build one dataset entry from a single sentence.

    The sentence is labeled first and validated against the schema. If the
    label is valid, paraphrase variants are generated and bundled together
    with the original sentence and label.

    Args:
        model: Name of the Ollama model used for labeling and paraphrasing.
        sentence: Command sentence to process.
        num_variants: Number of paraphrase variants to request.

    Returns:
        A dataset entry with ``sentence``, ``variants``, and ``label`` fields,
        or ``None`` when the generated label is invalid.
    """
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
    """Process a list of sentences in parallel.

    Each sentence is labeled and optionally paraphrased in a worker thread.
    Only valid samples are added to the final dataset.

    Args:
        sentences: List of input command sentences.
        model: Name of the Ollama model used for generation.
        num_variants: Number of paraphrase variants to request per sentence.
        workers: Number of parallel worker threads.

    Returns:
        A list of valid dataset entries.
    """
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
    """Verify dataset entries in parallel.

    Each sample is checked by an LLM. Samples that pass are returned in the
    verified list, while rejected samples are stored together with the model's
    explanation.

    Args:
        dataset: Generated dataset entries to verify.
        check_model: Name of the Ollama model used for verification.
        workers: Number of parallel worker threads for checking.

    Returns:
        A tuple ``(verified, rejected)``. ``verified`` contains accepted
        samples. ``rejected`` contains dictionaries with the sentence, label,
        and rejection reason.
    """
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
    """Save the generated dataset in raw and ChatML formats.

    The raw dataset is written as pretty-printed JSON. A second file is created
    as JSON Lines, where each line contains one ChatML training example derived
    from the dataset variants.

    Args:
        dataset: Dataset entries to save.
        save_path: Output path for the raw JSON dataset.
        tuning_path: Output path for the ChatML JSONL dataset.
    """
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
    """Run the full dataset generation workflow.

    The workflow loads input sentences, generates labels and paraphrases,
    optionally verifies the produced samples, and finally saves both accepted
    samples and fine-tuning data to disk.
    """
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
