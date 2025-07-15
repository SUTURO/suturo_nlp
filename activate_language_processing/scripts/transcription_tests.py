import sys
import warnings
import whisper  # Whisper model for transcription
import nlp_challenges  # Custom NLP utilities
from activate_language_processing.nlp import semanticLabelling  # Semantic labeling for intent/entity extraction
import spacy  # NLP library for processing text
from argparse import ArgumentParser  # Argument parsing for CLI
import rospy  # ROS Python client library
from pathlib import Path  # Path utilities for file handling

model = whisper.load_model("base")  # Load the Whisper model for transcription

# Load audio files from different conditions into separate lists
directory1 = Path("./AudioFiles2/Condition1/")
audios1 = [str(file) for file in directory1.glob("*") if file.is_file()]

directory2 = Path("./AudioFiles2/Condition2/")
audios2 = [str(file) for file in directory2.glob("*") if file.is_file()]

directory3 = Path("./AudioFiles2/Condition3/")
audios3 = [str(file) for file in directory3.glob("*") if file.is_file()]

directory4 = Path("./AudioFiles2/Condition4/")
audios4 = [str(file) for file in directory4.glob("*") if file.is_file()]

directory5 = Path("./AudioFiles2/Condition5/")
audios5 = [str(file) for file in directory5.glob("*") if file.is_file()]

directory6 = Path("./AudioFiles2/Condition6/")
audios6 = [str(file) for file in directory6.glob("*") if file.is_file()]

# Combine all audio file lists into a single list of folders
files = [audios1, audios2, audios3, audios4, audios5, audios6]


def transcribe_audio(audio_path):
    """
    Transcribe audio using Whisper model.
    
    Args:
        audio_path: Path to the audio file
    Returns: 
        Transcribed text
    """
    try:
        # Perform transcription using Whisper
        result = model.transcribe(audio_path, language="en")
        transcribed_text = result["text"].strip()
        #print(f"Transcribed text: {transcribed_text}")
        return transcribed_text
    except Exception as e:
        # Handle errors during transcription
        warnings.warn(f"Error during transcription: {e}")
        return None


def enhanced_transcription(audio_path, text):
    """
    Transcribe audio file again, but this time using a prompt providing context.

    Args:
        audio_path: Path to the audio file
        text: Initial transcribed text from the audio file
    Returns:
        Transcribed text with context
    """
    # Extract names and nouns from the initial transcription
    names, nouns = nlp_challenges.nounDictionary(text)

    # Generate a context-specific prompt based on extracted names and nouns
    if not names:
        prompt = f"The user wants to order one or several of these food or drink items: {' , '.join(nouns)}."
    elif not nouns:
        prompt = f"The user says their name is one of these: {' , '.join(names)}."
    else:
        prompt = f"The user says their name is one of these: {' , '.join(names)}. And they like to drink one of these: {' , '.join(nouns)}."

    # Perform transcription with the generated prompt
    result = model.transcribe(audio_path, initial_prompt=prompt)
    text = result["text"]
    #print(f"Enhanced transcription with context: {text.strip()}")
    return text


def compare_transcriptions(ground_truth, original_text, enhanced_text):
    """
    Compare original and enhanced transcriptions to a ground truth.
    
    Args:
        ground_truth: Ground truth text for comparison
        original_text: Original transcribed text
        enhanced_text: Enhanced transcribed text
    Returns:
        Comparison result as a string
    """
    # Compare both transcriptions to the ground truth and return the result
    if original_text == ground_truth and enhanced_text == ground_truth:
        return "Both transcriptions match the ground truth."
    elif original_text == ground_truth:
        return "Original transcription matches the ground truth, but enhanced does not."
    elif enhanced_text == ground_truth:
        return "Enhanced transcription matches the ground truth, but original does not."
    else:
        return "Neither transcription matches the ground truth."


def get_intent_and_entities(original_text, enhanced_text, context):
    """
    Extract intents and entities from original and enhanced transcriptions.

    Args:
        original_text: Original transcribed text
        enhanced_text: Enhanced transcribed text
        context: Context dictionary for semantic labeling
    Returns:
        Intents and entities for both original and enhanced transcriptions
    """
    # Perform semantic labeling on the original transcription
    original_parses = semanticLabelling(original_text, context)

    for p in original_parses:
        # Skip invalid parses
        if not p["sentence"].strip() or not p["entities"]:
            if (p["intent"] != 'affirm' and p["intent"] != "deny") or not p["sentence"].strip():
                rospy.loginfo(f"[ALP]: Skipping empty or invalid parse. Sentence: '{p['sentence']}', Intent: '{p['intent']}'")
                continue

        # Process entities for the original transcription
        pAdj1 = {"sentence": p["sentence"], "intent": p["intent"], "entities": []}
        for k, v in p["entities"].items():
            entity_data = v.copy()
            entity_data["role"] = v["role"]
            entity_data.pop("group")
            entity_data.pop("idx")
            pAdj1["entities"].append(entity_data)

    # Perform semantic labeling on the enhanced transcription
    enhanced_parses = semanticLabelling(enhanced_text, context)

    for p in enhanced_parses:
        # Skip invalid parses
        if not p["sentence"].strip() or not p["entities"]:
            if (p["intent"] != 'affirm' and p["intent"] != "deny") or not p["sentence"].strip():
                rospy.loginfo(f"[ALP]: Skipping empty or invalid parse. Sentence: '{p['sentence']}', Intent: '{p['intent']}'")
                continue

        # Process entities for the enhanced transcription
        pAdj2 = {"sentence": p["sentence"], "intent": p["intent"], "entities": []}
        for k, v in p["entities"].items():
            entity_data = v.copy()
            entity_data["role"] = v["role"]
            entity_data.pop("group")
            entity_data.pop("idx")
            pAdj2["entities"].append(entity_data)

    return pAdj1["intent"], pAdj1["entities"], pAdj2["intent"], pAdj2["entities"]


def compare_intents(ground_truth_intent, original_intent, enhanced_intent):
    """
    Compare intents from original and enhanced transcriptions to a ground truth.
    
    Args:
        ground_truth_intent: Ground truth intent
        original_intent: Intent from the original transcription
        enhanced_intent: Intent from the enhanced transcription
    Returns:
        Comparison result as a string
    """
    # Compare intents and return the result
    if original_intent == ground_truth_intent and enhanced_intent == ground_truth_intent:
        return "a"
    elif original_intent == ground_truth_intent and enhanced_intent != ground_truth_intent:
        return "b"
    elif enhanced_intent == ground_truth_intent and original_intent != ground_truth_intent:
        return "c"
    else:
        return "d"


def compare_entities(ground_truth_entities, original_entities, enhanced_entities):
    """
    Compare entities from original and enhanced transcriptions to a ground truth.
    
    Args:
        ground_truth_entities: Ground truth entities
        original_entities: Entities from the original transcription
        enhanced_entities: Entities from the enhanced transcription
    Returns:
        Comparison result as a string
    """
    # Compare entities and return the result
    if original_entities == ground_truth_entities and enhanced_entities == ground_truth_entities:
        return "a"
    elif original_entities == ground_truth_entities and enhanced_entities != ground_truth_entities:
        return "b"
    elif enhanced_entities == ground_truth_entities and original_entities != ground_truth_entities:
        return "c"
    else:
        return "d"


def test_multiple_files(files, context):
    """
    Test multiple audio files by transcribing them and comparing the results to ground truth.

    Args:
        files: List of lists containing paths to audio files
        context: Context dictionary for semantic labelling
    Returns:
        None
    """
    # Define ground truth examples for testing
    ground_truth1 = "my name is sarah and my favorite drink is coffee."
    ground_truth_intent1 = "Receptionist"
    ground_truth_entities1 = [ {'role': 'BeneficiaryRole', 'value': 'Sarah', 'entity': 'NaturalPerson', 'propertyAttribute': (), 'actionAttribute': (), 'numberAttribute': ()},
                              {'role': 'Item', 'value': 'coffee', 'entity': 'drink', 'propertyAttribute': (), 'actionAttribute': (), 'numberAttribute': ()}]

    groundtruth2 = "i would like to have two steaks, fries and cola."
    ground_truth_intent2 = "Order"
    ground_truth_entities2 = [{'role': 'Item', 'value': 'steaks', 'entity': 'food', 'propertyAttribute': (), 'actionAttribute': (), 'numberAttribute': ('two',)},
                               {'role': 'Item', 'value': 'fries', 'entity': 'food', 'propertyAttribute': (), 'actionAttribute': (), 'numberAttribute': ()}, 
                               {'role': 'Item', 'value': 'cola', 'entity': 'drink', 'propertyAttribute': (), 'actionAttribute': (), 'numberAttribute': ()}]
    groundtruth3 = "i like to play video games."
    ground_truth_intent3 = "Hobbies"
    ground_truth_entities3 = [{'role': 'Concept', 'value': 'video games', 'entity': 'Interest', 'propertyAttribute': (), 'actionAttribute': (), 'numberAttribute': ()}]


    # Initialize counters for correct predictions
    OriginalNameCount = 0
    OriginalOrderCount = 0
    OriginalHobbyCount = 0
    EnhancedNameCount = 0
    EnhancedOrderCount = 0
    EnhancedHobbyCount = 0
    counter = 1

    # Initialize lists to store false predictions
    falseOriginalName = []
    falseOriginalOrder = []
    falseOriginalHobby = []
    falseEnhancedName = []
    falseEnhancedOrder = []
    falseEnhancedHobby = []

    showCondition4 = []

    # Process each folder of audio files
    for folder in files:
        OriginalScore = 0
        EnhancedScore = 0
        OriginalIntentScore = 0
        EnhancedIntentScore = 0
        OriginalEntitiesScore = 0
        EnhancedEntitiesScore = 0
        # Process each audio file in the folder
        for audio in folder:
            original_text = transcribe_audio(audio)  
            original_text = original_text.strip().lower() 

            enhanced_text = enhanced_transcription(audio, original_text)
            enhanced_text = enhanced_text.strip().lower()

            original_intent, original_entities, enhanced_intent, enhanced_entities = get_intent_and_entities(original_text, enhanced_text, context)
            print(f"Enhanced entities: {enhanced_entities}")


            # Check audio files related to NameAndDrink
            if "NameAndDrink" in audio:
                check_intent = compare_intents(ground_truth_intent1, original_intent, enhanced_intent)
                check_entities = compare_entities(ground_truth_entities1, original_entities, enhanced_entities)

                # Compare transcriptions and update scores and counts
                if original_text == ground_truth1 and enhanced_text == ground_truth1:
                    OriginalScore += 1
                    EnhancedScore += 1
                    OriginalNameCount += 1
                    EnhancedNameCount += 1
                elif original_text == ground_truth1:
                    OriginalScore += 1
                    OriginalNameCount += 1
                    falseEnhancedName.append(enhanced_text + "\n")
                elif enhanced_text == ground_truth1:
                    EnhancedScore += 1
                    EnhancedNameCount += 1
                    falseOriginalName.append(original_text + "\n")
                else:
                    falseOriginalName.append(original_text + "\n")
                    falseEnhancedName.append(enhanced_text + "\n")
                
                # Update intent scores based on comparison
                if check_intent == "a":
                    OriginalIntentScore += 1
                    EnhancedIntentScore += 1
                elif check_intent == "b":
                    OriginalIntentScore += 1
                elif check_intent == "c":
                    EnhancedIntentScore += 1
                else:
                    pass

                # Update entity scores based on comparison
                if check_entities == "a":
                    OriginalEntitiesScore += 1
                    EnhancedEntitiesScore += 1
                elif check_entities == "b":
                    OriginalEntitiesScore += 1
                elif check_entities == "c":
                    EnhancedEntitiesScore += 1
                else:
                    pass
                
            # Check audio files related to Order
            if "Order" in audio:
                check_intent = compare_intents(ground_truth_intent2, original_intent, enhanced_intent)
                check_entities = compare_entities(ground_truth_entities2, original_entities, enhanced_entities)

                if original_text == groundtruth2 and enhanced_text == groundtruth2:
                    OriginalScore += 1
                    EnhancedScore += 1
                    OriginalOrderCount += 1
                    EnhancedOrderCount += 1
                elif original_text == groundtruth2:
                    OriginalScore += 1
                    OriginalOrderCount += 1
                    falseEnhancedOrder.append(enhanced_text + "\n")
                elif enhanced_text == groundtruth2:
                    EnhancedScore += 1
                    EnhancedOrderCount += 1 
                    falseOriginalOrder.append(original_text + "\n")
                else:
                    falseOriginalOrder.append(original_text + "\n")
                    falseEnhancedOrder.append(enhanced_text + "\n")

                # Update intent scores based on comparison
                if check_intent == "a":
                    OriginalIntentScore += 1
                    EnhancedIntentScore += 1
                elif check_intent == "b":
                    OriginalIntentScore += 1
                elif check_intent == "c":
                    EnhancedIntentScore += 1
                else:
                    pass

                # Update entity scores based on comparison
                if check_entities == "a":
                    OriginalEntitiesScore += 1
                    EnhancedEntitiesScore += 1
                elif check_entities == "b":
                    OriginalEntitiesScore += 1
                elif check_entities == "c":
                    EnhancedEntitiesScore += 1
                else:
                    pass
                
            # Check audio files related to Hobby
            if "Hobby" in audio:
                check_intent = compare_intents(ground_truth_intent3, original_intent, enhanced_intent)
                check_entities = compare_entities(ground_truth_entities3, original_entities, enhanced_entities)

                if original_text == groundtruth3 and enhanced_text == groundtruth3:
                    OriginalScore += 1
                    EnhancedScore += 1
                    OriginalHobbyCount += 1
                    EnhancedHobbyCount += 1
                elif original_text == groundtruth3:
                    OriginalScore += 1
                    OriginalHobbyCount += 1
                    falseEnhancedHobby.append(enhanced_text + "\n")
                elif enhanced_text == groundtruth3:
                    EnhancedScore += 1
                    EnhancedHobbyCount += 1
                    falseOriginalHobby.append(original_text + "\n")
                else:
                    falseOriginalHobby.append(original_text + "\n")
                    falseEnhancedHobby.append(enhanced_text + "\n")

                # Update intent scores based on comparison
                if check_intent == "a":
                    OriginalIntentScore += 1
                    EnhancedIntentScore += 1
                elif check_intent == "b":
                    OriginalIntentScore += 1
                elif check_intent == "c":
                    EnhancedIntentScore += 1
                else:
                    pass

                # Update entity scores based on comparison
                if check_entities == "a":
                    OriginalEntitiesScore += 1
                    EnhancedEntitiesScore += 1
                elif check_entities == "b":
                    OriginalEntitiesScore += 1
                elif check_entities == "c":
                    EnhancedEntitiesScore += 1
                else:
                    pass
                
        # Store scores and correct counts for each condition
        if counter == 1:
            OriginalCondition1 = OriginalScore
            EnhancedCondition1 = EnhancedScore
            OriginalCorrectIntent1 = OriginalIntentScore
            EnhancedCorrectIntent1 = EnhancedIntentScore
            OriginalCorrectEntities1 = OriginalEntitiesScore
            EnhancedCorrectEntities1 = EnhancedEntitiesScore
        elif counter == 2:
            OriginalCondition2 = OriginalScore
            EnhancedCondition2 = EnhancedScore
            OriginalCorrectIntent2 = OriginalIntentScore
            EnhancedCorrectIntent2 = EnhancedIntentScore
            OriginalCorrectEntities2 = OriginalEntitiesScore
            EnhancedCorrectEntities2 = EnhancedEntitiesScore
        elif counter == 3:
            OriginalCondition3 = OriginalScore
            EnhancedCondition3 = EnhancedScore
            OriginalCorrectIntent3 = OriginalIntentScore
            EnhancedCorrectIntent3 = EnhancedIntentScore
            OriginalCorrectEntities3 = OriginalEntitiesScore
            EnhancedCorrectEntities3 = EnhancedEntitiesScore
        elif counter == 4:
            OriginalCondition4 = OriginalScore
            EnhancedCondition4 = EnhancedScore
            OriginalCorrectIntent4 = OriginalIntentScore
            EnhancedCorrectIntent4 = EnhancedIntentScore
            OriginalCorrectEntities4 = OriginalEntitiesScore
            EnhancedCorrectEntities4 = EnhancedEntitiesScore
        elif counter == 5:
            OriginalCondition5 = OriginalScore
            EnhancedCondition5 = EnhancedScore
            OriginalCorrectIntent5 = OriginalIntentScore
            EnhancedCorrectIntent5 = EnhancedIntentScore
            OriginalCorrectEntities5 = OriginalEntitiesScore
            EnhancedCorrectEntities5 = EnhancedEntitiesScore
        elif counter == 6:
            OriginalCondition6 = OriginalScore
            EnhancedCondition6 = EnhancedScore
            OriginalCorrectIntent6 = OriginalIntentScore
            EnhancedCorrectIntent6 = EnhancedIntentScore
            OriginalCorrectEntities6 = OriginalEntitiesScore
            EnhancedCorrectEntities6 = EnhancedEntitiesScore

        counter += 1

    print(showCondition4)

    # Print results for each condition
    print(f"Correct OriginalCondition 1: {str(OriginalCondition1)}/{len(audios1)}")
    print(f"Correct OriginalCondition 2: {str(OriginalCondition2)}/{len(audios2)}")
    print(f"Correct OriginalCondition 3: {str(OriginalCondition3)}/{len(audios3)}")
    print(f"Correct OriginalCondition 4: {str(OriginalCondition4)}/{len(audios4)}")
    print(f"Correct OriginalCondition 5: {str(OriginalCondition5)}/{len(audios5)}")
    print(f"Correct OriginalCondition 6: {str(OriginalCondition6)}/{len(audios6)}")
    print(f"Original Condition 1 Intent Correct: {str(OriginalCorrectIntent1)}/{len(audios1)}")
    print(f"Original Condition 2 Intent Correct: {str(OriginalCorrectIntent2)}/{len(audios2)}")
    print(f"Original Condition 3 Intent Correct: {str(OriginalCorrectIntent3)}/{len(audios3)}")
    print(f"Original Condition 4 Intent Correct: {str(OriginalCorrectIntent4)}/{len(audios4)}")
    print(f"Original Condition 5 Intent Correct: {str(OriginalCorrectIntent5)}/{len(audios5)}")
    print(f"Original Condition 6 Intent Correct: {str(OriginalCorrectIntent6)}/{len(audios6)}")
    print(f"Original Condition 1 Entities Correct: {str(OriginalCorrectEntities1)}/{len(audios1)}")
    print(f"Original Condition 2 Entities Correct: {str(OriginalCorrectEntities2)}/{len(audios2)}")
    print(f"Original Condition 3 Entities Correct: {str(OriginalCorrectEntities3)}/{len(audios3)}")
    print(f"Original Condition 4 Entities Correct: {str(OriginalCorrectEntities4)}/{len(audios4)}")
    print(f"Original Condition 5 Entities Correct: {str(OriginalCorrectEntities5)}/{len(audios5)}")
    print(f"Original Condition 6 Entities Correct: {str(OriginalCorrectEntities6)}/{len(audios6)}")

    print(f"Correct EnhancedCondition 1: {str(EnhancedCondition1)}/{len(audios1)}")
    print(f"Correct EnhancedCondition 2: {str(EnhancedCondition2)}/{len(audios2)}")
    print(f"Correct EnhancedCondition 3: {str(EnhancedCondition3)}/{len(audios3)}")
    print(f"Correct EnhancedCondition 4: {str(EnhancedCondition4)}/{len(audios4)}")
    print(f"Correct EnhancedCondition 5: {str(EnhancedCondition5)}/{len(audios5)}")
    print(f"Correct EnhancedCondition 6: {str(EnhancedCondition6)}/{len(audios6)}")
    print(f"Enhanced Condition 1 Intent Correct: {str(EnhancedCorrectIntent1)}/{len(audios1)}")
    print(f"Enhanced Condition 2 Intent Correct: {str(EnhancedCorrectIntent2)}/{len(audios2)}")
    print(f"Enhanced Condition 3 Intent Correct: {str(EnhancedCorrectIntent3)}/{len(audios3)}")
    print(f"Enhanced Condition 4 Intent Correct: {str(EnhancedCorrectIntent4)}/{len(audios4)}")
    print(f"Enhanced Condition 5 Intent Correct: {str(EnhancedCorrectIntent5)}/{len(audios5)}")
    print(f"Enhanced Condition 6 Intent Correct: {str(EnhancedCorrectIntent6)}/{len(audios6)}")
    print(f"Enhanced Condition 1 Entities Correct: {str(EnhancedCorrectEntities1)}/{len(audios1)}")
    print(f"Enhanced Condition 2 Entities Correct: {str(EnhancedCorrectEntities2)}/{len(audios2)}")
    print(f"Enhanced Condition 3 Entities Correct: {str(EnhancedCorrectEntities3)}/{len(audios3)}")
    print(f"Enhanced Condition 4 Entities Correct: {str(EnhancedCorrectEntities4)}/{len(audios4)}")
    print(f"Enhanced Condition 5 Entities Correct: {str(EnhancedCorrectEntities5)}/{len(audios5)}")
    print(f"Enhanced Condition 6 Entities Correct: {str(EnhancedCorrectEntities6)}/{len(audios6)}")

    # Calculate and print overall correct counts
    totalOriginalCorrect = sum([OriginalCondition1,OriginalCondition2,OriginalCondition3,OriginalCondition4,OriginalCondition5,OriginalCondition6])
    totalOriginalCorrectIntent = sum([OriginalCorrectIntent1,OriginalCorrectIntent2,OriginalCorrectIntent3,OriginalCorrectIntent4,OriginalCorrectIntent5,OriginalCorrectIntent6])
    totalOriginalCorrectEntities = sum([OriginalCorrectEntities1,OriginalCorrectEntities2,OriginalCorrectEntities3,OriginalCorrectEntities4,OriginalCorrectEntities5,OriginalCorrectEntities6])
    totalEnhancedCorrect = sum([EnhancedCondition1,EnhancedCondition2,EnhancedCondition3,EnhancedCondition4,EnhancedCondition5,EnhancedCondition6])
    totalEnhancedCorrectIntent = sum([EnhancedCorrectIntent1,EnhancedCorrectIntent2,EnhancedCorrectIntent3,EnhancedCorrectIntent4,EnhancedCorrectIntent5,EnhancedCorrectIntent6])
    totalEnhancedCorrectEntities = sum([EnhancedCorrectEntities1,EnhancedCorrectEntities2,EnhancedCorrectEntities3,EnhancedCorrectEntities4,EnhancedCorrectEntities5,EnhancedCorrectEntities6])
    totalAudios = sum([len(audios1),len(audios2),len(audios3),len(audios4),len(audios5),len(audios6)])

    print(f"Overall correctly original identified audios: {totalOriginalCorrect}/{totalAudios}")
    print(f"Overall correctly original identified intents: {totalOriginalCorrectIntent}/{totalAudios}")
    print(f"Overall correctly original identified entities: {totalOriginalCorrectEntities}/{totalAudios}")
    print(f"Overall correctly enhanced identified audios: {totalEnhancedCorrect}/{totalAudios}")
    print(f"Overall correctly enhanced identified intents: {totalEnhancedCorrectIntent}/{totalAudios}")
    print(f"Overall correctly enhanced identified entities: {totalEnhancedCorrectEntities}/{totalAudios}")

    # Print correct counts per sentence type
    print("Correct Original per sentence")
    print(f"NameAndDrink: {str(OriginalNameCount)}/{totalOriginalCorrect}")
    print(f"Order: {str(OriginalOrderCount)}/{totalOriginalCorrect}")
    print(f"Hobby: {str(OriginalHobbyCount)}/{totalOriginalCorrect}")

    print("Correct Enhanced per sentence")
    print(f"NameAndDrink: {str(EnhancedNameCount)}/{totalEnhancedCorrect}")
    print(f"Order: {str(EnhancedOrderCount)}/{totalEnhancedCorrect}")
    print(f"Hobby: {str(EnhancedHobbyCount)}/{totalEnhancedCorrect}")
        
    # Print false transcriptions for analysis
    print("False Original Transcriptions:")
    for st in falseOriginalName:
        print(st)

    for st in falseOriginalOrder:
        print(st)
    
    for st in falseOriginalHobby:
        print(st)

    print("False Enhanced Transcriptions:")
    for st in falseEnhancedName:
        print(st)           
    for st in falseEnhancedOrder:
        print(st)
    for st in falseEnhancedHobby:
        print(st)


def main():
    """
    Main function to run the transcription and comparison process.

    Args:
        None
    Returns:
        None
    """
    # Parse command-line arguments
    parser = ArgumentParser(prog='activate_language_processing')
    parser.add_argument('-nlu', '--nluURI', default='http://localhost:5005/model/parse', help="Link towards the RASA semantic parser. Default: http://localhost:5005/model/parse")
    args, unknown = parser.parse_known_args(rospy.myargv()[1:])

    # Initialize context for semantic labeling
    rasaURI = args.nluURI
    intent2Roles = {}
    context = {
        "rasaURI": rasaURI,
        "nlp": spacy.load("en_core_web_sm"),
        "intent2Roles": intent2Roles,
        "role2Roles": {},
    }

    # Test multiple audio files
    test_multiple_files(files, context)


if __name__ == "__main__":
    # Suppress specific warnings
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
    main()
