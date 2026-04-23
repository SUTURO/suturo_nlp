import argparse
import json
from pathlib import Path

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
    "affirm",
    "lookup",
    "clarify",
]
ROLES = ["Person", "srcRoom", "destRoom", "Furniture", "Clothes", "Item"]
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
            "minItems": 1
        }
    },
}

EXAMPLE = json.dumps({
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
                    "numberAttribute": []
                },
                {
                    "role": "Furniture",
                    "value": "arm chair",
                    "entity": "DesignedFurniture",
                    "propertyAttribute": [],
                    "actionsAttribute": [],
                    "numberAttribute": []
                }
            ]
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
                    "numberAttribute": []
                },
                {
                    "role": "Furniture",
                    "value": "kitchen counter",
                    "entity": "DesignedFurniture",
                    "propertyAttribute": [],
                    "actionsAttribute": [],
                    "numberAttribute": []
                }
            ]
        }
    ]
}, indent=2)



# Load generated sentences from the official RoboCup@Home command generator
def load_sentences(path):
    with open(path, "r") as f:
        return [line.strip() for line in f]


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


def paraphrase_sentence(sentence, model):
    PROMPT = f"""
You are tasked with generating **one paraphrased versions** of a given task command.

Your input will be a **single task command**.
Your output must be **a single command** containing **the alternative phrasing** of that command **and nothing else**.

Given command:
{sentence}

---

### **Guidelines**

* **Content preservation:**

  * Keep all **entities, objects, and locations exactly the same** (e.g., “coke” must remain “coke”).
  * You may **restructure the sentence** as long as meaning and entities are preserved.

* **Tone and style:**

  * Maintain a **natural, conversational tone** write as if real people might say it.
  * Avoid robotic or overly formal phrasing unless required for the most complex version.    
"""

    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": PROMPT,
            },
        ],
        stream=False,
        options={"temperature": 0.9},
    )

    response = response.message.content.strip()
    return response


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, help="The model to use"),
    parser.add_argument(
        "-d", "--data", type=Path, required=True, help="List of the sentences"
    ),
    parser.add_argument(
        "-s", "--save", type=Path, required=True, help="Where to save the results (.json or .jsonl)"
    ),

    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    sentences = load_sentences(args.data)

    dataset = []
    try:
        for s in tqdm(sentences, desc="Processing sentences"):
            try:
                json_label = sentences_to_json(s, args.model)
                ok, err = json_validation(json_label)
                if not ok:
                    tqdm.write(f"Invalid sample skipped: {err}")
                    continue
                variants = [s]
                # Try to get three variants
                for _ in range(3):
                    try:
                        variants.append(paraphrase_sentence(s, args.model))
                    except:
                        pass

                dataset.append(
                    {
                        "sentence": s,
                        "variants": variants,
                        "label": json_label,
                    }
                )
            except Exception as e:
                tqdm.write(f"Error processing sentence: {s} {e}")
    except KeyboardInterrupt:
        print("Saving data...")
    finally:
        with open(args.save, "w") as f:
            json.dump(dataset, f, indent=2)

    print(f"Saved dataset with {len(dataset)} samples to {args.save}")


if __name__ == "__main__":
    main()
