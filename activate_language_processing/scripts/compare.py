from metaphone import doublemetaphone
from Levenshtein import distance as lev_dist
import yaml
from pathlib import Path
import re

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

allowed_entities = food + drink

def double_metaphone_similarity(word1, word2):
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


def testMetaphone(word1, word2):
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
    


dictionary = []
"""
print("Pronunciation Similarity Scores:")
for w in allowed_entities:
    if double_metaphone_similarity("slaves",w) >= 0.5:
        dictionary.append(f"Score {w}: {double_metaphone_similarity('slaves',w)}")
"""

print(double_metaphone_similarity("slaves", "steak"))

for e in dictionary:
    print(e)

print(len(allowed_entities))
print(len(dictionary))

