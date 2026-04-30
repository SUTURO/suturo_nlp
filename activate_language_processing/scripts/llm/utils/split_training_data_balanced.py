import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a roughly intent-balanced train/eval split."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("training_data.jsonl"),
        help="Input JSONL dataset in OpenAI messages format",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("training_data_train.jsonl"),
        help="Output train JSONL file",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=Path("training_data_eval.jsonl"),
        help="Output eval JSONL file",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Evaluation fraction, e.g. 0.1 for 10%% or 0.2 for 20%%",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    return parser.parse_args()


def extract_intents(line):
    """Extract the unique intents from one JSONL sample.

    Args:
        line: One JSONL line containing a training sample.

    Returns:
        tuple: Sorted unique intent names found in the assistant label.
    """
    row = json.loads(line)
    messages = row["messages"]
    label = json.loads(messages[-1]["content"])

    intents = set()
    for item in label["intents"]:
        intents.add(item["intent"])

    return tuple(sorted(intents))


def load_groups(path):
    """Group dataset rows by their intent combination.

    Args:
        path: Path to the input JSONL file.

    Returns:
        dict: Mapping from intent tuples to lists of JSONL lines.
    """
    groups = defaultdict(list)

    with open(path) as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            intents = extract_intents(line)
            groups[intents].append(line)

    return dict(groups)


def split_samples(groups, eval_ratio, seed):
    """Split grouped samples into train and eval sets.

    Args:
        groups: Dictionary of grouped samples by intent combination.
        eval_ratio: Fraction of each group that should go to eval.
        seed: Random seed used for shuffling.

    Returns:
        tuple: Two lists containing train samples and eval samples.
    """
    rng = random.Random(seed)

    train_samples = []
    eval_samples = []

    for group in groups.values():
        rng.shuffle(group)

        if len(group) == 1:
            n_eval = 0
        else:
            n_eval = round(len(group) * eval_ratio)
            n_eval = max(1, n_eval)
            n_eval = min(len(group) - 1, n_eval)

        eval_samples.extend(group[:n_eval])
        train_samples.extend(group[n_eval:])

    rng.shuffle(train_samples)
    rng.shuffle(eval_samples)
    return train_samples, eval_samples


def write_jsonl(path, samples):
    """Write JSONL samples to a file.

    Args:
        path: Output file path.
        samples: List of JSONL lines to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        for sample in samples:
            file.write(sample + "\n")


def print_distribution(name, samples):
    """Print the intent distribution for a dataset split.

    Args:
        name: Label used in the printed output.
        samples: List of JSONL lines in the split.
    """
    counts = Counter()
    for line in samples:
        for intent in extract_intents(line):
            counts[intent] += 1

    total = counts.total()
    print(f"{name}: {len(samples)} samples, {total} labeled intents")
    for intent in sorted(counts):
        pct = 100.0 * counts[intent] / total if total else 0.0
        print(f"  {intent:20s} {counts[intent]:5d}  {pct:6.2f}%")


def main():
    args = parse_args()
    if not 0.0 < args.eval_ratio < 0.5:
        raise ValueError("--eval-ratio must be between 0 and 0.5")

    groups = load_groups(args.input)
    train_samples, eval_samples = split_samples(groups, args.eval_ratio, args.seed)
    if not train_samples or not eval_samples:
        raise ValueError("Split failed: one of the splits is empty")

    write_jsonl(args.train_output, train_samples)
    write_jsonl(args.eval_output, eval_samples)

    print(f"Input: {args.input}")
    print(f"Train: {len(train_samples)} -> {args.train_output}")
    print(f"Eval:  {len(eval_samples)} -> {args.eval_output}")
    print_distribution("Train intent distribution", train_samples)
    print_distribution("Eval intent distribution", eval_samples)


if __name__ == "__main__":
    main()
