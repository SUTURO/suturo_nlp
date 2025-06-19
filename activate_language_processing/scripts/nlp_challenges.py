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

# Load the entities.yml file from our rasa model
with open('/home/siwall/ros/nlp_ws/src/suturo_rasa/entities.yml', 'r') as file:
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

def nounDictionary(text):
    """
    Takes the the whisper transcription and extracts the NOUNs and PROPNs (the relevant entities). And
    checks wheter they appear in the list of entities of the rasa model we use. If not, the given term is replaced 
    in the text.

    Args:
        text: The whisper result

    Returns:
        A list of words that are likely the entities the user uttered.
    """    
    
    # Process the text
    doc = nlp(text)

    # Collect all unique terms that need replacement
    names_to_replace = set()
    nouns_to_replace = set()


    # Extract named entities (multi-word)
    named_ents = {ent.text: ent for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "FAC"}}
    for text_entity in named_ents:
        if text_entity not in allowed_entities:
            if text_entity not in nouns_to_replace:
                # If the entity is not in the allowed entities, add it to names_to_replace
                names_to_replace.add(text_entity)
                print(f"names to replace {names_to_replace}")

    # Go through individual tokens (words) in the text
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"}:
            if token.text not in allowed_entities and token.text[:-1] not in allowed_entities:
                if token.text not in names_to_replace:
                    nouns_to_replace.add(token.text)
                    print(f"nouns to replace: {nouns_to_replace}")

    names = []
    dictionary = []

    p = inflect.engine()

    def pluralize(word):
        # Check if the word is already plural
        if p.singular_noun(word):
            return word  # It's already plural
        else:
            return p.plural(word)  # Convert to plural

    print(f"names to replace: {names_to_replace}")
    print(f"nouns to replace: {nouns_to_replace}")

    def fill_names(names_to_replace, allowed_entities, threshold, max_attempts=16):
        if not names_to_replace:
            return 
        attempts = 0
        while attempts < max_attempts:
            for term in names_to_replace:
                for entity in allowed_entities:
                    if double_metaphone_similarity(pluralize(term), pluralize(entity)) >= threshold:
                        if pluralize(entity) not in names and pluralize(entity) not in dictionary:  # Avoid duplicates in names
                            names.append(pluralize(entity))
            if len(names) >= 5:  # Ensure we have at least 3 valid entities
                break
            threshold -= 0.05  # Reduce threshold for next attempt
            attempts += 1

    def fill_dictionary(nouns_to_replace, allowed_entities, threshold, max_attempts=16):
        if not nouns_to_replace:
            return 
        attempts = 0
        while attempts < max_attempts:
            for term in nouns_to_replace:
                for entity in allowed_entities:
                    if double_metaphone_similarity(pluralize(term), pluralize(entity)) >= threshold:
                        if pluralize(entity) not in dictionary and pluralize(entity) not in names:  # Avoid duplicates in names
                            dictionary.append(pluralize(entity))
            if len(dictionary) >= 5:  # Ensure we have at least 3 valid entities
                break
            threshold -= 0.05  # Reduce threshold for next attempt
            attempts += 1

    fill_dictionary(nouns_to_replace, allowed_entities, threshold=0.9)
    fill_names(names_to_replace, allowed_entities, threshold=0.9)

    return names, dictionary


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

                #print(value)
                number = ent.get("numberAttribute")
                    
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


