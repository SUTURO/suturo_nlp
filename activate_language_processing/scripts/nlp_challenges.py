import json
from word2number import w2n
<<<<<<< Updated upstream
import re

=======
from typing import List
from pydantic import BaseModel
import yaml
import spacy
from pathlib import Path
import librosa
#from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch
import ast
import numpy as np
#from metaphone import doublemetaphone
#from Levenshtein import distance as lev_dist
import re


# Load processor and model
#processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

"""
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    device_map={"": "cpu"},
    torch_dtype=torch.float32  # Safe for CPU
)
"""

def replace_text(text, audio):
    """
    Takes the the whisper transcription and extracts the NOUNs and PROPNs (the relevant entities). And
    checks wheter they appear in the list of entities of the rasa model we use. If not, the given term is replaced 
    in the text.

    Args:
        text: The whisper result

    Returns:
        The modified text.
    """    
    
    # Load the entities.yml file from our rasa model
    with open('/home/suturo/ros_ws/nlp_ws/src/suturo_rasa/entities.yml', 'r') as file:
        data = yaml.safe_load(file)

    # Create separate lists for our entities
    food = data.get('food', {}).get('entities', [])
    drink = data.get('drink', {}).get('entities', [])
    clothing = data.get('Clothing', {}).get('entities', [])
    furniture = data.get('DesignedFurniture', {}).get('entities', [])
    people = data.get('NaturalPerson', {}).get('entities', [])
    rooms = data.get('Room', {}).get('entities', [])
    transportable = data.get('Transportable', {}).get('entities', [])
    interests = data.get('Interest', {}).get('entities', [])


    # The entities that are in our rasa model and are thus valid
    allowed_entities = food + drink
    allowed_entities.append('name')
    allowed_entities.append('drink')
    allowed_entities.append('food')

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Process the text
    doc = nlp(text)

    # Collect all unique terms that need replacement
    terms_to_replace = set()

    # Extract named entities (multi-word)
    named_ents = {ent.text: ent for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "FAC"}}
    for text_entity in named_ents:
        if text_entity not in allowed_entities:
            terms_to_replace.add(text_entity)

    # Go through individual tokens (words) in the text
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"}:
            if token.text not in allowed_entities and token.text[:-1] not in allowed_entities:
                terms_to_replace.add(token.text)

    torch.cuda.empty_cache()

    replacement_results = {term: replace_term2(term, [w for w in allowed_entities if double_metaphone_similarity(term, w) >= 0.5],audio) for term in terms_to_replace}
            
    # Call replace_term2 once for all unique terms
    #replacement_results = {term: replace_term2(term, allowed_entities, audio) for term in terms_to_replace}

    # Build replacement map using precomputed replacements
    replacement_map = {}
    for text_entity, ent in named_ents.items():
        if text_entity in replacement_results:
            replacement_map[(ent.start_char, ent.end_char)] = replacement_results[text_entity]

    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and token.text in replacement_results:
            print(f"False Noun: {token}")
            replacement_map[(token.idx, token.idx + len(token))] = replacement_results[token.text]

    # Start from the end of the text to avoid offset issues while replacing
    new_text = text
    for (start_char, end_char), replacement in sorted(replacement_map.items(), key=lambda x: -x[0][0]):
        # Replace the span from start_char to end_char with the replacement term
        new_text = new_text[:start_char] + replacement + new_text[end_char:]

    # Output the final modified text
    print(f"New text: {new_text}")
    return new_text


def double_metaphone_similarity(word1, word2):
    """
    Takes two strings and returns the similarity in terms of pronounciations as a value between 0 (not at all similar)
    and 1 (very similar).

    Args:
        word1/word2: A word (string) which similiarity in pronounciation is to be compared to a different word.

    Returns:
        The absolute similarity between word1 and word2 in terms of pronounciation.
    """
    # 1. Normalize inputs and handle multi-word terms
    def preprocess(word):
        word = word.lower().strip()
        # Remove non-alphabetic characters and split into words
        words = re.findall(r'[a-z]+', word)
        return words
    
    words1 = preprocess(word1)
    words2 = preprocess(word2)
    
    # Short-circuit for identical words
    if ' '.join(words1) == ' '.join(words2):
        return 1.0
    
    # 2. Get Double Metaphone codes for all words
    def get_metaphone(word):
        primary, secondary = doublemetaphone(word)
        return (primary or "", secondary or "")
    
    # Get codes for all words in each term
    meta1 = [get_metaphone(w) for w in words1]
    meta2 = [get_metaphone(w) for w in words2]
    
    # 3. Calculate best phonetic similarity between word pairs
    def phonetic_score(meta1, meta2):
        best_score = 0.0
        
        # 1. Calculate word-count difference penalty
        word_count_diff = abs(len(meta1) - len(meta2))
        word_count_penalty = max(0.6, 1 - (word_count_diff / 3))  # Penalty scales with difference (cap at 40% penalty)
        
        # 2. Compare individual word pairs
        for m1 in meta1:
            for m2 in meta2:
                # Primary code comparison
                if m1[0] and m2[0]:
                    max_len = max(len(m1[0]), len(m2[0]))
                    raw_score = 1 - (lev_dist(m1[0], m2[0]) / max_len)
                    best_score = max(best_score, raw_score)
             
                # Secondary code comparison
                if m1[1] and m2[1]:
                    max_len = max(len(m1[1]), len(m2[1]))
                    raw_score = 1 - (lev_dist(m1[1], m2[1]) / max_len)
                    best_score = max(best_score, raw_score)

        
        # 3. Apply word-count penalty
        return best_score * word_count_penalty
    
    phonetic_sim = phonetic_score(meta1, meta2)
    
    
    # 4. Improved syllable counting
    def count_syllables(word):
        word = re.sub(r'[^a-z]', '', word.lower())
        # Count vowel groups, but don't count silent 'e' at end
        if len(word) > 1 and word.endswith('e'):
            word = word[:-1]
        syllables = len(re.findall(r'[aeiouy]+', word))
        return max(1, syllables)
    
    # Calculate average syllables per word
    syl1 = sum(count_syllables(w) for w in words1) / len(words1)
    syl2 = sum(count_syllables(w) for w in words2) / len(words2)
    syllable_sim = 1 - (abs(syl1 - syl2) / max(syl1, syl2, 1))
    
    # 5. Length difference penalty (now compares total lengths)
    len1 = sum(len(w) for w in words1)
    len2 = sum(len(w) for w in words2)
    len_diff_penalty = 1 - (abs(len1 - len2) / (max(len1, len2) * 0.75))  # More lenient
    
    # 6. Combined score with adjusted weights
    combined_score = (
        0.75 * phonetic_sim +  # Increased weight to phonetic match
        0.15 * syllable_sim +  # Reduced weight to syllable count
        0.1 * len_diff_penalty
    )
    
    # 7. Additional penalty for multi-word mismatches
    if len(words1) != len(words2):
        combined_score *= 0.8  # Penalize if different number of words
    
    return min(1.0, max(0.0, combined_score))


def replace_term2(flase_term, entities, audio):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    # Handle both path and waveform inputs
    if isinstance(audio, (str, Path)):
        # Case 1: Audio file path
        audio_path = Path(audio) if isinstance(audio, str) else audio
        sr = processor.feature_extractor.sampling_rate
        audio_waveform, _ = librosa.load(audio_path, sr=sr)
    elif isinstance(audio, np.ndarray):
        # Case 2: Pre-loaded waveform
        audio_waveform = audio
        sr = processor.feature_extractor.sampling_rate
    elif hasattr(audio, 'get_wav_data'):  # speech_recognition.AudioData
        # Case 3: speech_recognition AudioData object
        wav_bytes = audio.get_wav_data()
        wav_array = np.frombuffer(wav_bytes, dtype=np.int16)
        # Skip WAV header if present (first 44 bytes)
        if len(wav_array) > 22:  # Rough check for header
            wav_array = wav_array[22:]
        audio_waveform = librosa.util.buf_to_float(wav_array, dtype=np.float32)
        sr = 16000  # Default for speech_recognition
    else:
        raise ValueError("Unsupported audio input type")

    # Construct prompt
    prompt = (
        "You will be given a term, list of entities and an audio file. "
        "Please select an entity from the list that sounds closest to the given term when pronounced and return it in json format. You must not return anything that is not in the given list of entities."
        "You may also not return the original given term. For example: the term 'wet boy' should be replaced with 'red bull'. "
        f"Entities: {', '.join(entities)}. Term: " + flase_term
    )

    prompt2 = (
        "Please transcribe the file I will give you"
    )

    # Build structured conversation
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "local"},  # Placeholder
            {"type": "text", "text": prompt}
        ]}
    ]

    # Generate chat prompt with <|AUDIO|> token
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    # Process input (new correct usage)
    inputs = processor(
        text=text,
        audio=audio_waveform,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate response
    torch.cuda.empty_cache()
    generate_ids = model.generate(**inputs, max_new_tokens=13)
    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

    del inputs, generate_ids
    torch.cuda.empty_cache()

    try:
        result = ast.literal_eval(response)
        entity_value = result.get("entity", "")
        print("Extracted entity:", entity_value)
        return entity_value
    except (ValueError, SyntaxError) as e:
        print("Failed to parse response:", response)
        entity_value = ""
        return entity_value


>>>>>>> Stashed changes
def switch(case, response, context):
    '''
    Manual Implementation of switch(match)-case because python3.10 first implemented one, this uses 3.8.
    
    Args:
        case: The intent parsed from the response
        response: The formatted .json from the record function

    Returns:
        The function corresponding to the intent
    '''
    return {
        "Receptionist": lambda: Receptionist.receptionist(response,context),
        "Order": lambda: Restaurant.order(response,context),
        "Hobbies": lambda: Receptionist.hobbies(response,context),
        "affirm": lambda: context["pub"].publish(f"<CONFIRM>, True"),
        "deny": lambda: context["pub"].publish(f"<DENY>, False")
    }.get(case, lambda: context["pub"].publish(f"<NONE>"))()

def replace_word_and_next(text, target_word, replacement):
    # Regular expression to find the target word followed by another word
    pattern = rf"\b{target_word}\s+\w+\b"
    return re.sub(pattern, replacement, text)

"""
def is_number(value):
    
    Check if a given string is a number (either numeral or word).

    Args:
        value: a string, that might be a digit or word representation of a number.
    
    Returns:
        True if the string is a textual representation of a number (or a digit) else False.
    
    try:
     
        w2n.word_to_num(value)  
        return True
    except ValueError:
        return value.isdigit() 
"""

"""        
def to_number(value):
    
    Converts a number in words or numerals to an integer.

    Args:
        value: A string that contains a digit or word representation of a number.
        
    Returns:
        The input number as an integer.
    
    return int(value) if value.isdigit() else w2n.word_to_num(value)
"""

blacklist = {"states": "steaks", "slates": "steaks", "slaves": "steaks", "stakes": "steaks", 
        "red boy": "red bull", "redbull": "red bull", "whetball": "red bull", "whet ball": "red bull",
        "red bullseye": "red bull", "red balloon": "red bull", "red bullet": "red bull", "bed pull": "red bull",
        "let bull": "red bull", "wet bull": "red bull","dead bull": "red bull","red boot": "red bull","red bell": "red bull",
        "red pool": "red bull","red bowl": "red bull","read bull": "red bull","red pull": "red bull","red ball": "red bull",
        "rad bull": "red bull","rat bull": "red bull","red full": "red bull","red wool": "red bull","rip bull": "red bull",
        "wetball": "red bull", "wet ball": "red bull", "wet": "red bull", "boy": "red bull"}

def getData(response):
        """
        Function for extracting names, drinks, and foods from entities in a JSON response.

        Args:
            response: The JSON string containing entities.
        
        Returns:
            A JSON string categorizing names, drinks, and foods.
        """
        
        # Parse the response JSON string into a dictionary
        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")

        drinks = []
        foods = []
        names = []
        interests = []

        cachedNumer = 0
        falseNumber = False

        entities = response_dict.get("entities", [])
        if not isinstance(entities, list):
            raise ValueError("Expected 'entities' to be a list in the response.")

        # Filtering the entities list for drink, food and NaturalPerson and the amount of each
        for ent in entities:
            if isinstance(ent, dict):
                entity = ent.get("entity")
                value = ent.get("value")
                value = value.strip()

                if value == "boy":
                    entity = "drink"

                if value in blacklist:
                    value = blacklist.get(value)

                #print(value)
                number = ent.get("numberAttribute")
                
                """
                if is_number(value):
                    cachedNumer = to_number(value)
                    falseNumber = True
                    continue
                """
                    
                if entity == "drink":
                    if not number:
                        drinks.append((value, 1))
                    else:
                        drinks.append((value, number[0] if type(number[0]) == int else w2n.word_to_num(number[0])))
                elif entity == "food":
                    if not number:
                        foods.append((value,1))
                    else:
                        foods.append((value, number[0] if type(number[0]) == int else w2n.word_to_num(number[0])))
                elif entity == "NaturalPerson":
                    names.append(value)
                elif entity == "Interest":
                    interests.append(value)
                
        # Build the .json
        values= {"names": names, "drinks": drinks, "foods": foods, "hobbies": interests}
        return json.dumps(values) 



class Receptionist:
    """
    A class for the Receptionist challenge. Extracts the entities and their values and checks
    whether they are a NaturalPerson, a Drink or Food. In the end a string is published
    containing the name and the favorite drink of a person.

    Methods:
        receptionist(response, context):
            Processes the data given by getData and publishes a string
            with the extracted NaturalPerson and drink.
        hobbies(response, context):
            Processes the data given by getData and publishes a string
            with the extracted Interest.
    """
    
    def receptionist(response,context):
        '''
        Function for the receptionist task. Extracts the first name and drink in the list of names 
        and drinks the getData() function may return and publishes those. 

        Args:
            response: Formatted .json from record function.
            context:
                pub: a ROS publisher object to publish processed results to a specified topic.
        '''
        data = json.loads(getData(response)) 

        name = data.get("names")
        if name:
            name = name[0]
            
        drink = data.get("drinks")
        drink = [x[0] for x in drink][0] if drink else None

        context["pub"].publish(f"<GUEST>; {name}; {drink}")

    def hobbies(response, context):
        '''
        Function to extract interests from the list of hobbies the getData() function may return
        and publishes those.

        Args:
            response: Formatted .json from record function.
            context:
                pub: a ROS publisher object to publish processed results to a specified topic.
        '''
        data = json.loads(getData(response))

        hobby = data.get("hobbies")

        context["pub"].publish(f"<INTERESTS>; {hobby}")


class Restaurant:
    """
    A class for the Restaurant challenge. Extracts the entities and their values and checks
    whether they are a NaturalPerson, a Drink or Food. In the end a string is published
    containing the food and drink a person woukd like to order.

    Methods:
        restaurant(response, context):
            Processes the date given by getData and publishes a string with the extracted NaturalPerson and Drink.
    """

    def order(response, context):
        """
        Function for the order task.

        Args:
            response: JSON string with entities from `nluInternal`.
            context: Context dictionary, including:
                pub: ROS publisher object to publish results to a specified topic.
        """
        data = json.loads(getData(response))

        food = data.get("foods")

        drink = data.get("drinks")

        # Publish the order
        context["pub"].publish(f"<ORDER>, {food}, {drink}")


