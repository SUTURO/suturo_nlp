import argparse
import csv
import json
import subprocess
import threading
import time
import timeit
from dataclasses import asdict, dataclass
from pathlib import Path

from ollama import chat
from tqdm import tqdm

DEFAULT_MODELS = ["NLP-gemma3", "NLP-llama31", "NLP-qwen25", "NLP-qwen35", "NLP-ministral"]


@dataclass
class VRAMResult:
    baseline_mb: int | None
    peak_mb: int | None
    delta_mb: int | None


@dataclass
class Metrics:
    intent_f1: float
    entity_f1: float
    overall_f1: float


@dataclass
class BenchmarkResult:
    id: int | str
    model: str
    category: str
    input: str
    valid_json: bool
    time_s: float
    error: str | None
    vram: VRAMResult
    metrics: Metrics


def norm(v):
    return str(v).strip().lower()


def f1_sets(expected: set, predicted: set) -> float:
    tp = len(expected & predicted)
    p = tp / len(predicted) if predicted else (1.0 if not expected else 0.0)
    r = tp / len(expected) if expected else (1.0 if not predicted else 0.0)
    return round((2 * p * r / (p + r)) if (p + r) else 0.0, 4)


def extract_intents(parsed) -> list:
    if not isinstance(parsed, dict):
        return []
    intents = parsed.get("intents", [])
    return intents if isinstance(intents, list) else []


def score_intents(exp_intents: list, pred_intents: list) -> float:
    exp = {norm(i.get("intent", "")) for i in exp_intents if isinstance(i, dict)}
    pred = {norm(i.get("intent", "")) for i in pred_intents if isinstance(i, dict)}
    exp.discard("")
    pred.discard("")
    return f1_sets(exp, pred)


def score_entities(exp_intents: list, pred_intents: list) -> float:
    """
    Match entities on (intent, role, value, entity type).

    Attributes are ignored because they vary a lot and are harder to compare
    fairly. Intent is included so an entity in the wrong intent is not counted
    as correct.
    """

    def flatten(intents):
        entities = set()
        for intent in intents:
            if not isinstance(intent, dict):
                continue
            intent_name = norm(intent.get("intent", ""))
            for e in intent.get("entities", []):
                if not isinstance(e, dict):
                    continue
                entities.add(
                    (
                        intent_name,
                        norm(e.get("role", "")),
                        norm(e.get("value", "")),
                        norm(e.get("entity", "")),
                    )
                )
        return entities

    return f1_sets(flatten(exp_intents), flatten(pred_intents))


def get_vram_mb():
    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3)
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

    total = 0
    for line in out.stdout.splitlines():
        try:
            total += int(line.strip().split(",")[0])
        except ValueError:
            pass
    return total


def sample_vram(stop_event, samples, interval_s):
    while not stop_event.is_set():
        used = get_vram_mb()
        if used is not None:
            samples.append(used)
        time.sleep(interval_s)


def run(
    test_cases: list, models: list, measure_vram: bool, vram_interval: float
) -> list:
    results = []
    total = len(models) * len(test_cases)
    progress = tqdm(total=total, desc="Benchmarking", unit="case")

    for model in models:
        tqdm.write(f"\n-- {model} {'-' * 45}")
        for idx, tc in enumerate(test_cases, start=1):
            case_id = tc.get("id", idx)
            inp = tc["input"]
            category = tc.get("category", "unknown")
            exp_intents = tc.get("expected", {}).get("intents", [])

            baseline_vram = get_vram_mb() if measure_vram else None
            vram_samples = []
            stop_event = threading.Event()
            sampler = None

            if measure_vram:
                sampler = threading.Thread(
                    target=sample_vram,
                    args=(stop_event, vram_samples, vram_interval),
                    daemon=True,
                )
                sampler.start()

            raw = ""
            error = None
            t0 = timeit.default_timer()
            try:
                resp = chat(
                    model=model,
                    messages=[{"role": "user", "content": inp}],
                    think=False,
                    format="json",
                )
                raw = resp.message.content.strip()
            except Exception as ex:
                error = str(ex)
            elapsed = round(timeit.default_timer() - t0, 3)

            if sampler is not None:
                stop_event.set()
                sampler.join(timeout=2)

            try:
                parsed = json.loads(raw) if raw else None
                valid_json = parsed is not None
            except json.JSONDecodeError:
                parsed = None
                valid_json = False

            pred_intents = extract_intents(parsed)
            if valid_json:
                intent_f1 = score_intents(exp_intents, pred_intents)
                entity_f1 = score_entities(exp_intents, pred_intents)
            else:
                intent_f1 = 0.0
                entity_f1 = 0.0
            overall_f1 = round((intent_f1 + entity_f1) / 2.0, 4)

            vram_peak = max(vram_samples) if vram_samples else None
            vram_delta = (
                vram_peak - baseline_vram
                if vram_peak is not None and baseline_vram is not None
                else None
            )

            progress.update(1)
            progress.set_postfix(
                model=model,
                id=case_id,
                json=valid_json,
                f1=f"{overall_f1:.2f}",
                time=f"{elapsed}s",
                refresh=False,
            )

            tqdm.write(
                f"  [id={case_id} {category}] json={valid_json} "
                f"intent={intent_f1:.2f} entity={entity_f1:.2f} "
                f"time={elapsed}s vram_delta={vram_delta if vram_delta is not None else 'n/a'}"
            )

            results.append(
                BenchmarkResult(
                    id=case_id,
                    model=model,
                    category=category,
                    input=inp,
                    valid_json=valid_json,
                    time_s=elapsed,
                    error=error,
                    vram=VRAMResult(baseline_vram, vram_peak, vram_delta),
                    metrics=Metrics(intent_f1, entity_f1, overall_f1),
                )
            )
    progress.close()
    return results


def build_summary(results: list) -> list:
    summary = []
    for model in sorted({r.model for r in results}):
        rows = [r for r in results if r.model == model]
        n = len(rows)
        deltas = [r.vram.delta_mb for r in rows if r.vram.delta_mb is not None]

        summary.append(
            {
                "model": model,
                "samples": n,
                "valid_json_rate": round(sum(r.valid_json for r in rows) / n, 4),
                "mean_time_s": round(sum(r.time_s for r in rows) / n, 4),
                "mean_intent_f1": round(sum(r.metrics.intent_f1 for r in rows) / n, 4),
                "mean_entity_f1": round(sum(r.metrics.entity_f1 for r in rows) / n, 4),
                "mean_overall_f1": round(
                    sum(r.metrics.overall_f1 for r in rows) / n, 4
                ),
                "mean_vram_delta_mb": (
                    round(sum(deltas) / len(deltas), 2) if deltas else None
                ),
                "peak_vram_delta_mb": max(deltas) if deltas else None,
            }
        )
    return summary


def flatten_results(results: list) -> list:
    rows = []
    for r in results:
        rows.append(
            {
                "id": r.id,
                "model": r.model,
                "category": r.category,
                "input": r.input,
                "valid_json": r.valid_json,
                "time_s": r.time_s,
                "error": r.error,
                "intent_f1": r.metrics.intent_f1,
                "entity_f1": r.metrics.entity_f1,
                "overall_f1": r.metrics.overall_f1,
                "vram_baseline_mb": r.vram.baseline_mb,
                "vram_peak_mb": r.vram.peak_mb,
                "vram_delta_mb": r.vram.delta_mb,
            }
        )
    return rows


def write_csv(path: str, rows: list):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data", required=True, help="Path to ground-truth JSONL test file"
    )
    parser.add_argument("-m", "--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("-s", "--save", default="benchmark_results.json")
    parser.add_argument("--no-vram", action="store_true", help="Disable VRAM tracking")
    parser.add_argument("--vram-interval", type=float, default=0.05)
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        test_cases = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(test_cases)} test cases | models: {args.models}")
    results = run(
        test_cases, args.models, not args.no_vram, max(0.02, args.vram_interval)
    )
    summary = build_summary(results)

    payload = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }

    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    save_path = Path(args.save)
    summary_csv = save_path.with_name(f"{save_path.stem}_summary.csv")
    detailed_csv = save_path.with_name(f"{save_path.stem}_detailed.csv")
    write_csv(str(summary_csv), summary)
    write_csv(str(detailed_csv), flatten_results(results))

    print(f"\nSaved -> {args.save}")
    print(f"Saved -> {summary_csv}")
    print(f"Saved -> {detailed_csv}")


if __name__ == "__main__":
    main()
