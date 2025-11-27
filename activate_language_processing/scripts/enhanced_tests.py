# TODO:
#       1. Able to test single Intents
#       2. Some sentences are classified false, even tough they are correct
#          Example:
#               My name is Sarah (Correct)
#               My name is sara (incorrect)
#           In our case this is not completely false.
#       3. Metrics for each category and intent / entities.
#       4. Something like F1-Score for entities.
#               Right now if one entity is false, the entire list is false.
#       5. Documentation

import warnings
from argparse import ArgumentParser
from pathlib import Path
import jiwer
import nlp_challenges
import spacy
import whisper
import yaml
from activate_language_processing.nlp import semanticLabelling
from rclpy.logging import get_logger
import pandas as pd
import requests
from tabulate import tabulate
import numpy as np

WHISPER_MODEL = "base"
AUDIO_FILES_DIRECTORY = "./AudioFiles"
GROUND_TRUTH_FILE = "./references.yml"
RESULT_FILE = "./results.json"

# rclpy logger
logger = get_logger("TEST")


def check_rasa(uri):
    try:
        payload = {"text": "Hello World!"}
        logger.info(f"Checking Rasa URI: {uri}")
        response = requests.post(uri, json=payload, timeout=5)

        if response.status_code == 200:
            logger.info("Rasa is available.")
            return True
    except requests.exceptions.ConnectionError:
        raise ConnectionError
    finally:
        logger.error("RASA is not available. Maybe you forgot to start Rasa!")
        return False


def load_whisper():
    """
    Loads the whisper model
    Returns:
        Whisper model
    """
    logger.info(f"Loading whisper model ({WHISPER_MODEL})...")
    model = whisper.load_model(WHISPER_MODEL)
    logger.info("Done.")
    return model


def load_reference():
    """
    Load all references (Ground Truth) from the `references.yml` file.
    Returns:
        Dictionary of ground truth references
    """
    try:
        with open(GROUND_TRUTH_FILE, "r") as file:
            data = yaml.safe_load(file)
            # We only want the GT
            return data["ground_truth"]
    except FileNotFoundError as e:
        logger.error(e)
        return None


def load_audio_files():
    """
    List all .wav audio files from the given directory.
    Returns:
        Dictionary with a list of .wav audio files of every condition.
    """
    path = Path(AUDIO_FILES_DIRECTORY)
    if not path.exists():
        raise FileNotFoundError(
            f"Audio files directory not found: {AUDIO_FILES_DIRECTORY}"
        )
    conditions = {}
    file_sum = 0
    for directory in path.glob("*"):
        if directory.is_dir():
            # TODO: Sorting the list?
            # Get a list of all .wav files from every condition
            audio_files = [
                str(file) for file in directory.glob("*.wav") if file.is_file()
            ]
            if not audio_files:
                logger.error(f"No wav files found in directory: {directory}.")
            else:
                file_sum += len(audio_files)
                logger.info(f"Found {len(audio_files)} audio files in {directory}.")
                conditions[directory.name] = audio_files
    logger.info(f"OVERALL {file_sum} FILES FOUND.")
    return conditions


def transcribe_normal(model, path):
    """
    Transcribe the given wav audio file.
    Args:
        model: Whisper model
        path: Path to .wav file
    Returns:
        Audio transcription as string
    """
    try:
        result = model.transcribe(path, language="en")
        return result["text"].strip()
    except Exception as e:
        logger.error(f"Could not transcribe audio: {path} [EXCEPTION]: {e}")
        return None


def transcribe_enhanced(model, path, orig_transcription):
    """
    Transcribe audio using Whisper. Here we're using a prompt providing context.
    Args:
        model: Whisper model
        path: Path to .wav file
        orig_transcription: Original transcription
    Returns:
        Enhanced audio transcription as string
    """
    try:
        names, nouns = nlp_challenges.nounDictionary(orig_transcription)
        if not names:
            prompt = f"The user wants to order one or several food pr drinks items: {' , '.join(nouns)}."
        elif not nouns:
            prompt = f"The user says their name is one of these: {' , '.join(names)}."
        else:
            prompt = f"The user says their name is one of these: {' , '.join(names)}. And they likes to drink one of these: {' , '.join(nouns)}."
        result = model.transcribe(path, initial_prompt=prompt, language="en")
        return result["text"].strip()
    except Exception as e:
        logger.error(f"Could not transcribe audio with prompt: {path} [EXCEPTION]: {e}")
        return None


def get_intent_and_entities(original_transcription, enhanced_transcription, context):
    """
    Extract intent and entities from the given transcription.
    Args:
        original_transcription: Original transcription
        enhanced_transcription: Enhanced transcription
        context: Context for semantic labeling
    Returns:
        Tuple with intent and list of entities
    """

    def _handle_parses(parses):
        for p in parses:
            if not p["sentence"].strip() or (
                not p["entities"] and p["intent"] not in ["affirm", "deny"]
            ):
                continue
            pAdj = {
                "sentence": p["sentence"],
                "intent": p["intent"],
                "entities": [],
            }
            for _, v in p["entities"].items():
                entity_data = v.copy()
                entity_data["role"] = v["role"]
                entity_data.pop("group")
                entity_data.pop("idx")
                pAdj["entities"].append(entity_data)
            return pAdj
        return logger.error(f"Empty parse: {parses}")

    original_parses = semanticLabelling(original_transcription, context)
    enhanced_parses = semanticLabelling(enhanced_transcription, context)
    orig = _handle_parses(original_parses)
    enh = _handle_parses(enhanced_parses)
    # Return the result as Tuple
    return orig["intent"], orig["entities"], enh["intent"], enh["entities"]


def get_audio_category(filename):
    """
    Extract the category from the audio file.
    Example:
        Hobby1.1.wav --> Hobby1
    Returns:
        Category name from an audio file.
    """
    # Remove the extension and get the first index integer from the filename
    return Path(filename).stem.split(".")[0]


def compare_intent(ground_truth, transcription):
    return ground_truth == transcription


def compare_entities(ground_truth, transcription):
    if len(ground_truth) != len(transcription):
        return False

    def _normalize(items):
        l = []
        for item in items:
            l.append(
                (item.get("role", ""), item.get("value", ""), item.get("entity", ""))
            )
        return sorted(l)

    return _normalize(ground_truth) == _normalize(transcription)


def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Ratio (WER).
    Args:
        reference: Reference transcription
        hypothesis: Hypothesis transcription
    Returns:
        WER
    """
    return jiwer.wer(reference, hypothesis)


def test_file(model, ground_truth, file, context):
    condition = Path(file).parent.name
    filename = Path(file).name
    category = get_audio_category(file)

    if category not in ground_truth:
        logger.error(f"Category {category} not found!")
        return None

    ground_truth_text = ground_truth[category].get("text", "")
    ground_truth_intent = ground_truth[category].get("intent", "")
    ground_truth_entities = ground_truth[category].get("entities", [])

    normal_transcription = transcribe_normal(model, file)
    enhanced_transcription = transcribe_enhanced(model, file, normal_transcription)

    normal_intent, normal_entities, enhanced_intent, enhanced_entities = (
        get_intent_and_entities(normal_transcription, enhanced_transcription, context)
    )

    wer_normal = calculate_wer(ground_truth_text, normal_transcription)
    wer_enh = calculate_wer(ground_truth_text, enhanced_transcription)

    is_correct_entities_normal = compare_entities(
        ground_truth_entities, normal_entities
    )
    is_correct_entities_enh = compare_entities(ground_truth_entities, enhanced_entities)

    is_correct_intent_normal = compare_intent(ground_truth_intent, normal_intent)
    is_correct_intent_enhanced = compare_intent(ground_truth_intent, enhanced_intent)

    return {
        "Condition": condition,
        "Filename": filename,
        "Category": category,
        "Ground_Truth": ground_truth_text,
        "Normal_Transcription": normal_transcription,
        "Enhanced_Transcription": enhanced_transcription,
        "WER_Normal": wer_normal,
        "WER_Enhanced": wer_enh,
        "Ground_Truth_Intent": ground_truth_intent,
        "Normal_Transcription_Intent": normal_intent,
        "Enhanced_Transcription_Intent": enhanced_intent,
        "Correct_Intent_Normal": is_correct_intent_normal,
        "Correct_Intent_Enhanced": is_correct_intent_enhanced,
        "Ground_Truth_Entities": ground_truth_entities,
        "Normal_Transcription_Entities": normal_entities,
        "Enhanced_Transcription_Entities": enhanced_entities,
        "Correct_Entities_Normal": is_correct_entities_normal,
        "Correct_Entities_Enhanced": is_correct_entities_enh,
    }


def print_results(df):
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80 + "\n")
    print(f"FILES TESTED: {len(df)}\n")

    # Metrics
    intent_acc_normal = df["Correct_Intent_Normal"].mean()
    intent_acc_enhanced = df["Correct_Intent_Enhanced"].mean()

    entities_acc_normal = df["Correct_Entities_Normal"].mean()
    entities_acc_enhanced = df["Correct_Entities_Enhanced"].mean()

    wer_normal_avg = df["WER_Normal"].mean()
    wer_enhanced_avg = df["WER_Enhanced"].mean()

    summary = pd.DataFrame(
        {
            "Metric": ["Intent Match", "Entities Match", "WER (avg)"],
            "Normal": [
                f"{intent_acc_normal:.1%} ({df['Correct_Intent_Normal'].sum()}/{len(df)})",
                f"{entities_acc_normal:.1%} ({df['Correct_Entities_Normal'].sum()}/{len(df)})",
                f"{wer_normal_avg:.3f}",
            ],
            "Enhanced": [
                f"{intent_acc_enhanced:.1%} ({df['Correct_Intent_Enhanced'].sum()}/{len(df)})",
                f"{entities_acc_enhanced:.1%} ({df['Correct_Entities_Enhanced'].sum()}/{len(df)})",
                f"{wer_enhanced_avg:.3f}",
            ],
        }
    )

    print(tabulate(summary, headers="keys", tablefmt="outline", showindex=False))
    print("\n" + "=" * 80)
    print(f"MORE DETAILS IN: {RESULT_FILE}")
    print("=" * 80 + "\n")


def save_results(dataframe):
    dataframe.to_json(RESULT_FILE, indent=2, orient="index")


def test_all_files(model, refs, audio_files, context):
    results = []

    for _, files in audio_files.items():
        for file in files:
            row = test_file(model, refs, file, context)
            if row is not None:
                results.append(row)

    df = pd.DataFrame(results)

    save_results(df)
    print_results(df)


def main():
    parser = ArgumentParser(prog="activate_language_processing")
    parser.add_argument(
        "-nlu",
        "--nluURI",
        default="http://localhost:5005/model/parse",
        help="Link towards the RASA semantic parser. Default: http://localhost:5005/model/parse",
    )
    args = parser.parse_args()
    context = {
        "rasaURI": args.nluURI,
        "nlp": spacy.load("en_core_web_sm"),
        "intent2Roles": {},
        "role2Roles": {},
    }
    check_rasa(context["rasaURI"])
    model = load_whisper()
    refs = load_reference()
    audio_files = load_audio_files()
    test_all_files(model, refs, audio_files, context)


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", message="FP16 is not supported on CPU; using FP32 instead"
    )
    main()
