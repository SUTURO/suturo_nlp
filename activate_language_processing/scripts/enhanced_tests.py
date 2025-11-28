# TODO:
#       1. Some sentences are classified false, even tough they are correct
#          Example:
#               My name is Sarah (Correct)
#               My name is sara (incorrect)
#           In our case this is not completely wrong.
#       2. Metrics for each category and intent / entities.
#       3. Something like F1-Score for entities.
#               Right now if one entity is wrong, the entire list is wrong.
#       4. Add result summary to json file
#       5. Collect all wrong sentences and store them somewhere

import warnings
from argparse import ArgumentParser
from pathlib import Path
import jiwer
import spacy
import whisper
import yaml
import pandas as pd
from rclpy.logging import get_logger
import requests
from tabulate import tabulate
# import numpy as np

from activate_language_processing.nlp import semanticLabelling
import nlp_challenges


WHISPER_MODEL = "base"
AUDIO_FILES_DIRECTORY = "./AudioFiles"
GROUND_TRUTH_FILE = "./references.yml"
RESULT_FILE = "./results.json"

# rclpy logger
logger = get_logger("TEST")


def check_rasa(uri):
    """
    Check if rasa is available, before we start testing.

    Args:
        uri: The rasa URI we use if we start rasa.

    Returns:
        None

    Raises:
        ConnectionError if rasa is not available.
    """
    try:
        payload = {"text": "Hello World!"}
        logger.info(f"Checking Rasa URI: {uri}")
        response = requests.post(uri, json=payload, timeout=5)

        if response.status_code == 200:
            logger.info("Rasa is available.")
            return
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Rasa is not available. Maybe you forgot to start Rasa!")


def load_whisper():
    """
    Loads the whisper model.

    Returns:
        Whisper model
    """
    logger.info(f"Loading whisper model ({WHISPER_MODEL})...")
    model = whisper.load_model(WHISPER_MODEL)
    logger.info("Done.")
    return model


def load_reference():
    """
    Load all references (Ground Truth) from the ``references.yml`` file.

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

    Example:
        {"Condition1": ["Hobby1.1.wav", "Hobby1.2.wav", ..."]}.

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


def get_specific_intent(audio_files, intent):
    """
    Filter audio files to only include categories matching a given intent prefix.

    Args:
        audio_files: Dictionary of condition and list of audio files.
        intent: The intent prefix to filter or None if not given.

    Example:
        {"Condition1": ["Hobby1.1.wav", "Hobby1.2.wav", ..."]}.

    Returns:
        A dictionary containing only the intent and the files whose intent start with the given prefix.
        If no intent prefix is given, the dictionary with all files will be returned.
    """
    if not intent:
        return audio_files

    matching_files = {}

    for condition, files in audio_files.items():
        f_list = [file for file in files if get_audio_category(file).startswith(intent)]
        if f_list:
            matching_files[condition] = f_list
    total = sum(len(files) for files in matching_files.values())
    logger.info(f"OVERALL {total} FILES FOUND WITH MATCHING INTENT.")
    return matching_files


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
        entities = []
        for item in items:
            entities.append(
                (item.get("role", ""), item.get("value", ""), item.get("entity", ""))
            )
        return sorted(entities)

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
    """
    Test an audio file and compare with ground truth.

    Args:
        model: Whisper model
        ground_truth: The ground truth from ``references.yml``
        file: The audio file
        context: Rasa context

    Returns:
        Dictionary with all relevant test results.
    """
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


def print_results(df, intent, intent_gpr_table=None):
    """
    Print all relevant test results in terminal.

    Args:
        intent: Intent prefix
        df: Dataframe with test results.

    Returns:
        None
    """
    # Metrics

    # Intents
    intent_acc_normal = df["Correct_Intent_Normal"].mean()
    intent_acc_enhanced = df["Correct_Intent_Enhanced"].mean()

    # Entities
    entities_acc_normal = df["Correct_Entities_Normal"].mean()
    entities_acc_enhanced = df["Correct_Entities_Enhanced"].mean()

    # WER
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

    # Intent Table

    intent_grp_table = (
        df.groupby("Ground_Truth_Intent")
        .agg(
            Correct_Normal=("Correct_Intent_Normal", "sum"),
            Correct_Enhanced=("Correct_Intent_Enhanced", "sum"),
            Total=("Ground_Truth_Intent", "count"),
        )
        .reset_index()
    )

    intent_grp_table["Normal"] = intent_grp_table.apply(
        lambda row: f"{row['Correct_Normal'] / row['Total']:.1%} ({row['Correct_Normal']}/{row['Total']})",
        axis=1,
    )
    intent_grp_table["Enhanced"] = intent_grp_table.apply(
        lambda row: f"{row['Correct_Enhanced'] / row['Total']:.1%} ({row['Correct_Enhanced']}/{row['Total']})",
        axis=1,
    )
    intent_grp_table = intent_grp_table[["Ground_Truth_Intent", "Normal", "Enhanced"]]

    # Entities Table

    expl_df = df.explode("Ground_Truth_Entities")
    # Convert dicts to tuples so they can be grouped
    expl_df["Ground_Truth_Entities"] = expl_df["Ground_Truth_Entities"].apply(
        lambda x: (x.get("role", ""), x.get("value", ""), x.get("entity", ""))
    )

    entities_grp_table = (
        expl_df.groupby("Ground_Truth_Entities")
        .agg(
            Correct_Normal=("Correct_Entities_Normal", "sum"),
            Correct_Enhanced=("Correct_Entities_Enhanced", "sum"),
            Total=("Ground_Truth_Entities", "count"),
        )
        .reset_index()
    )

    entities_grp_table["Normal"] = entities_grp_table.apply(
        lambda row: f"{row['Correct_Normal'] / row['Total']:.1%} ({row['Correct_Normal']}/{row['Total']})",
        axis=1,
    )
    entities_grp_table["Enhanced"] = entities_grp_table.apply(
        lambda row: f"{row['Correct_Enhanced'] / row['Total']:.1%} ({row['Correct_Enhanced']}/{row['Total']})",
        axis=1,
    )
    entities_grp_table = entities_grp_table[
        ["Ground_Truth_Entities", "Normal", "Enhanced"]
    ]

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80 + "\n")
    print(f"FILES TESTED: {len(df)}")
    if not intent:
        print("INTENT TESTED: All\n")
    else:
        print(f"INTENT TESTED: {intent}\n")
    print(tabulate(summary, headers="keys", tablefmt="outline", showindex=False))

    print("\n" + "-" * 80)
    print("INTENTS:\n")
    print(
        tabulate(intent_grp_table, headers="keys", tablefmt="outline", showindex=False)
    )

    print("\n" + "-" * 80)
    print("ENTITIES:\n")
    print(
        tabulate(
            entities_grp_table, headers="keys", tablefmt="outline", showindex=False
        )
    )

    print("\n" + "=" * 80)
    print(f"MORE DETAILS IN: {RESULT_FILE}")
    print("=" * 80 + "\n")


def save_results(dataframe):
    """
    Save test results to as json file.

    Args:
        dataframe: Dataframe with test results.

    Returns:
        None
    """
    dataframe.to_json(RESULT_FILE, indent=2, orient="index")


def test_all_files(model, refs, audio_files, context, intent):
    """
    Test alle audio files using ``test_file()``.

    Args:
        model: Whisper model
        refs: Ground truth from ``references.yml``
        audio_files: Dictionary of audio files
        context: Rasa context

    Returns:
        None
    """
    results = []

    for _, files in audio_files.items():
        for file in files:
            result = test_file(model, refs, file, context)
            if result is not None:
                results.append(result)

    df = pd.DataFrame(results)

    save_results(df)
    print_results(df, intent)


def main():
    parser = ArgumentParser(prog="activate_language_processing")
    parser.add_argument(
        "-nlu",
        "--nluURI",
        default="http://localhost:5005/model/parse",
        help="Link towards the RASA semantic parser. Default: http://localhost:5005/model/parse",
    )
    parser.add_argument(
        "-i",
        "--intent",
        default=None,
        help="Only test a specific intent (e.g. 'Order')",
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
    audio_files = get_specific_intent(audio_files, args.intent)
    test_all_files(model, refs, audio_files, context, args.intent)


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", message="FP16 is not supported on CPU; using FP32 instead"
    )
    main()
