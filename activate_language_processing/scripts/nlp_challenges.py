import json
from word2number import w2n
from typing import List
from pydantic import BaseModel
import yaml
import spacy
from pathlib import Path
#import librosa
#import torch
#import ast
import numpy as np
from metaphone import doublemetaphone
from Levenshtein import distance as lev_dist
import re
import inflect

home = Path.home()

# Load the entities.yml file from our rasa model
entities_file_path = Path(__file__).resolve().parents[3] / 'suturo_rasa' / 'entities.yml'
with open(entities_file_path, 'r') as file:
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
allowed_entities = people + food + drink
allowed_entities.append('name')
allowed_entities.append('drink')
allowed_entities.append('food')
allowed_entities.append('I')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
entity_labels = {"PERSON", "ORG", "GPE", "LOC", "FAC"}
p = inflect.engine()

def parse_doc(text, nlp):
    '''
    Prases the input text with spaCy.
    Args:
        text: Input whisper text
        nlp: spaCy NLP pipeline instance.

    Returns: spaCy Doc object containing token,POS tags and entities.

    '''
    return nlp(text)

def collect_named_entities(doc, allowed_entities):
    '''
    Collects named entities from the document that are not in the allowed entities list.
    Args:
        doc: Parsed NLP document
        allowed_entities: List of allowed entities.

    Returns:
        result: Set of named entity strings extracted from the document.
    '''
    result = set ()
    for ent in doc.ents:
        if ent.label_ in entity_labels:
            if ent.text not in allowed_entities:
                result.add(ent.text)
    return result

def collect_nouns(doc, allowed_entities, excluded_term):
    '''
    Collects nouns and proper nouns from a spaCy document.
    Args:
        doc: spaCy Doc object.
        allowed_entities:  Set of entities that should be ignored.
        excluded_term:  Terms that should be excluded.

    Returns:
        result: Set of noun strings extracted from the document.
    '''

    result = set ()
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"}:
            base = token.text.rstrip("s")
            if base not in allowed_entities and token.text not in excluded_term:
                result.add(token.text)
    return result

def pluralize(word):
    '''
    Converts a word into its plural form.
    Args:
        word: Input word

    Returns:
        word: Pluralized word if applicable.

    '''
    return word if p.singular_noun(word) else p.plural(word)

def pluralize_term(term):
    '''
    Pluralizes the last word of a term.
    Args:
        term:  Word or multi-word term.

    Returns:
        term: Term with the last word pluralized.
    '''
    words = term.split()

    if len(words) == 1:
        return pluralize(words[0])

    # only pluralize the only word.
    words[-1] = pluralize(words[-1])
    return " ".join(words)

def phonic_match_terms(terms,allowed_entities, similarity_fn, threshold=0.9,min_result=5, max_attempts=16):
    '''
    Matches input terms to allowed entities using a phonic similarity funtion.
    Args:
        terms: Terms to be matched.
        allowed_entities: List of allowed target entities.
        similarity_fn:Function used to calculate similarity.
        threshold: Initial similarity threshold.
        min_result: Minimum number of matches required.
        max_attempts: Maximum number of threshold adjustments.

    Returns:
        match: List of matched entities.
    '''

    match = []
    current_threshold = threshold

    for _ in range(max_attempts):
        for term in terms:
            normalized = pluralize_term(term)

            for entity in allowed_entities:
                score = similarity_fn(normalized, entity)
                if score >= current_threshold and entity not in match:
                    match.append(entity)
        if len(match) >= min_result:
            break
        current_threshold -= 0.05

    return match

def nounDictionary(text):
    '''
    Extract names and nouns from the text and matches them to allowed entities.
    Args:
        text: Input text to be analyzed

    Returns:
        names: List of matched person names.
        dictionary: List of matched dictionary entities.

    '''

    doc = parse_doc(text,nlp)

    names_to_replace = collect_named_entities(doc, allowed_entities)
    nouns_to_replace = collect_nouns(doc, allowed_entities, excluded_term=names_to_replace)
    names = phonic_match_terms(names_to_replace, allowed_entities, double_metaphone_similarity)
    dictionary = phonic_match_terms(nouns_to_replace, allowed_entities, double_metaphone_similarity)

    return names, dictionary

# --- Phonetic Similarity Calculation ---

# Constants for scoring
WORD_COUNT_PENALTY_CAP = 0.6
WORD_COUNT_PENALTY_DIVISOR = 3.0
LENGTH_DIFF_PENALTY_FACTOR = 0.75
PHONETIC_SIM_WEIGHT = 0.75
SYLLABLE_SIM_WEIGHT = 0.15
LENGTH_DIFF_WEIGHT = 0.1
MULTI_WORD_MISMATCH_PENALTY = 0.8

def preprocess_words(text: str) -> List[str]:
    '''
    Normalizes and splits text into individual words.
    Args:
        text: Input text.

    Returns:
        words: List of processed words.

    '''

    text = text.lower().strip()
    return re.findall(r'[a-z]+', text)

def get_metaphone_codes(word):
    '''
    Computes the metaphone code for a word.
    Args:
        word:Input word

    Returns:
        codes: Tuple containing primary and secondary metaphone codes.

    '''
    primary, secondary = doublemetaphone(word)
    return (primary or "", secondary or "")

def calculate_phonetic_score(words1, words2):
    '''
        Calculates the phonetic  similarity score  between two word lists.

    Args:
        words1: First list of words.
        words2: Second list of words.

    Returns:
        score: Phonetic similarity score.
    '''
    meta1 = [get_metaphone_codes(w) for w in words1]
    meta2 = [get_metaphone_codes(w) for w in words2]
    best_score = 0.0

    for m1 in meta1:
        for m2 in meta2:
            # Primary code comparison
            if m1[0] and m2[0]:
                max_len = max(len(m1[0]), len(m2[0]))
                if max_len > 0:
                    raw_score = 1 - (lev_dist(m1[0], m2[0]) / max_len)
                    best_score = max(best_score, raw_score)
         
            # Secondary code comparison
            if m1[1] and m2[1]:
                max_len = max(len(m1[1]), len(m2[1]))
                if max_len > 0:
                    raw_score = 1 - (lev_dist(m1[1], m2[1]) / max_len)
                    best_score = max(best_score, raw_score)

    word_count_diff = abs(len(meta1) - len(meta2))
    word_count_penalty = max(WORD_COUNT_PENALTY_CAP, 1 - (word_count_diff / WORD_COUNT_PENALTY_DIVISOR))
    
    return best_score * word_count_penalty

def count_syllables(word):
    '''
    Counts the number of syllables in a word.
    Args:
        word: Input word.

    Returns:
        syllable: Number of syllables in a word.

    '''
    word = re.sub(r'[^a-z]', '', word.lower())
    if len(word) > 1 and word.endswith('e'):
        word = word[:-1]
    syllables = len(re.findall(r'[aeiouy]+', word))
    return max(1, syllables)

def calculate_syllable_similarity(words1, words2):
    '''
    Calculates the syllable similarity between two lists of words.
    Args:
        words1: First list of words.
        words2: Second list of words.

    Returns:
        syllable_similarity: Syllable similarity score.

    '''
    if not words1 or not words2:
        return 0.0
    
    avg_syl1 = sum(count_syllables(w) for w in words1) / len(words1)
    avg_syl2 = sum(count_syllables(w) for w in words2) / len(words2)
    
    max_avg_syl = max(avg_syl1, avg_syl2, 1)
    return 1 - (abs(avg_syl1 - avg_syl2) / max_avg_syl)

def calculate_length_penalty(words1, words2):
    '''
    Calculates the length of each syllable in a word.
    Args:
        words1: Friendly list of words.
        words2: Second list of words.

    Returns:
        penalty: Length difference penalty value.

    '''
    len1 = sum(len(w) for w in words1)
    len2 = sum(len(w) for w in words2)
    max_len = max(len1, len2)
    
    if max_len == 0:
        return 1.0
        
    return 1 - (abs(len1 - len2) / (max_len * LENGTH_DIFF_PENALTY_FACTOR))

def double_metaphone_similarity(word1: str, word2: str) -> float:
    """
    Calculates a composite similarity score between two words based on phonetics, syllables, and length.
    The score is a value between 0 (not at all similar) and 1 (identical).

    Args:
        word1: The first word or phrase.
        word2: The second word or phrase.

    Returns:
        The composite similarity score between 0 and 1.
    """
    words1 = preprocess_words(word1)
    words2 = preprocess_words(word2)

    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    if ' '.join(words1) == ' '.join(words2):
        return 1.0

    phonetic_sim = calculate_phonetic_score(words1, words2)
    syllable_sim = calculate_syllable_similarity(words1, words2)
    len_diff_penalty = calculate_length_penalty(words1, words2)

    combined_score = (
            PHONETIC_SIM_WEIGHT * phonetic_sim +
            SYLLABLE_SIM_WEIGHT * syllable_sim +
            LENGTH_DIFF_WEIGHT * len_diff_penalty
    )

    if len(words1) != len(words2):
        combined_score *= MULTI_WORD_MISMATCH_PENALTY

    return min(1.0, max(0.0, combined_score))


def switch(case, response, context):
    '''
    Dispatches a case to the correct function based on the intent.

    This function acts as a router, calling the appropriate handler based on the
    `case` (intent) provided. It parses the response data once and passes it
    to the handlers, improving performance and adhering to DRY and SRP principles.
    
    Args:
        case: The intent parsed from the response.
        response: The formatted JSON from the recorder function.
        context: A dictionary containing contextual information, like a ROS publisher.

    Returns:
        The result of the called function.
    '''
    try:
        data = extract_entities(response)
    except (json.JSONDecodeError, ValueError) as e:
        context["pub"].publish(f"<ERROR_PARSING_RESPONSE>")
        return

    # Map intents to their handler functions
    handler_map = {
        "Receptionist": receptionist,
        "Order": restaurant,
        "Hobbies": receptionist,
        "affirm": lambda d, c: c["pub"].publish(f"<CONFIRM>, True"),
        "deny": lambda d, c: c["pub"].publish(f"<DENY>, False")
    }

    # Get the handler function for the given case, or a default handler
    handler = handler_map.get(case, lambda d, c: c["pub"].publish(f"<NONE>"))
    
    # Execute the handler
    return handler(data, context)

def replace_word_and_next(text, target_word, replacement):
    # Regular expression to find the target word followed by another word
    pattern = rf"\b{target_word}\s+\w+\b"
    return re.sub(pattern, replacement, text)

def extract_entities(response_string):
    """
    Parses a JSON response string and extracts relevant entities into a structured dictionary.
    This function has a single responsibility: converting the NLU JSON into a clean Python dict.

    Args:
        response_string: The JSON string containing NLU entities.
    
    Returns:
        A dictionary categorizing names, drinks, foods, and hobbies.
    
    Raises:
        ValueError: If the response is not a valid JSON or the structure is unexpected.
    """
    response_dict = json.loads(response_string)

    drinks = []
    foods = []
    names = []
    interests = []

    entities = response_dict.get("entities", [])
    if not isinstance(entities, list):
        raise ValueError("Expected 'entities' to be a list in the response.")

    # Filtering the entities list for drink, food and NaturalPerson and the amount of each
    for ent in entities:
        if isinstance(ent, dict):
            entity = ent.get("entity")
            value = ent.get("value")
            if not value:
                continue
            value = value.strip()

            number = ent.get("numberAttribute")
                
            if entity == "drink":
                if not number:
                    drinks.append((value, 1))
                else:
                    drinks.append((value, number[0] if isinstance(number[0], int) else w2n.word_to_num(number[0])))
            elif entity == "food":
                if not number:
                    foods.append((value,1))
                else:
                    foods.append((value, number[0] if isinstance(number[0], int) else w2n.word_to_num(number[0])))
            elif entity == "NaturalPerson":
                names.append(value)
            elif entity == "Interest":
                interests.append(value)
            
    return {"names": names, "drinks": drinks, "foods": foods, "hobbies": interests}

# --- Intent Handlers ---

def receptionist(data: dict, context: dict):
    '''
    Handles the receptionist guest task. Extracts name and drink from parsed data and publishes them.

    Args:
        data: Parsed dictionary from extract_entities().
        context: Execution context, containing the ROS publisher.
    '''
    name = data.get("names")
    name = name[0] if name else None
        
    drink = data.get("drinks")
    drink = drink[0][0] if drink else None # Get name of first drink

    context["pub"].publish(f"<GUEST>; {name}; {drink}")

def receptionist(data: dict, context: dict):
    '''
    Handles the receptionist hobbies task. Extracts hobbies and publishes them.

    Args:
        data: Parsed dictionary from extract_entities().
        context: Execution context, containing the ROS publisher.
    '''
    hobby = data.get("hobbies")
    # The original code published the list, so we'll keep that behavior.
    context["pub"].publish(f"<INTERESTS>; {hobby}")

def restaurant(data: dict, context: dict):
    """
    Handles the restaurant order task. Extracts food and drinks and publishes the order.

    Args:
        data: Parsed dictionary from extract_entities().
        context: Execution context, containing the ROS publisher.
    """
    food = data.get("foods")
    drink = data.get("drinks")

    # The original code published the lists of tuples, we preserve that.
    context["pub"].publish(f"<ORDER>, {food}, {drink}")


