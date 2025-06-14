from metaphone import doublemetaphone
from Levenshtein import distance as lev_dist
import yaml
from pathlib import Path
import re

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

allowed_entities = food + drink

def double_metaphone_similarity(word1, word2):
    # 1. Normalize inputs
    word1 = word1.lower().strip()
    word2 = word2.lower().strip()
    
    # Short-circuit for identical words
    if word1 == word2:
        return 1.0
    
    # 2. Get Double Metaphone codes
    def get_metaphone(word):
        primary, secondary = doublemetaphone(word)
        return (primary or "", secondary or "")
    
    meta1 = get_metaphone(word1)
    meta2 = get_metaphone(word2)
    
    # 3. Calculate phonetic similarity
    def phonetic_score(m1, m2):
        scores = []
        for code1 in m1:
            for code2 in m2:
                if not code1 or not code2:
                    continue
                max_len = max(len(code1), len(code2))
                score = 1 - (lev_dist(code1, code2) / max_len)
                scores.append(score)
        return max(scores) if scores else 0.0
    
    phonetic_sim = phonetic_score(meta1, meta2)
    
    # 4. Calculate syllable similarity (approximate)
    def count_syllables(word):
        word = re.sub(r'[^a-z]', '', word.lower())
        syllables = re.findall(r'[aeiouy]+', word)
        return max(1, len(syllables))
    
    syl1 = count_syllables(word1)
    syl2 = count_syllables(word2)
    syllable_sim = 1 - (abs(syl1 - syl2) / max(syl1, syl2, 1))
    
    # 5. Length difference penalty
    len_diff_penalty = 1 - (abs(len(word1) - len(word2)) / max(len(word1), len(word2), 1))
    
    # 6. Combined score with weights
    combined_score = (
        0.6 * phonetic_sim +  # Primary weight to phonetic match
        0.3 * syllable_sim +  # Secondary weight to syllable count
        0.1 * len_diff_penalty  # Small weight to length similarity
    )
    
    # 7. Apply minimum threshold
    if phonetic_sim < 0.7:  # If phonetic match isn't strong
        combined_score *= 0.7  # Penalize the score
    
    return min(1.0, max(0.0, combined_score))  # Clamp between 0-1

dictionary = []

print("Pronunciation Similarity Scores:")
for w in allowed_entities:
    if double_metaphone_similarity("slaves",w) >= 0.2:
        dictionary.append(f"Score {w}: {double_metaphone_similarity('slaves',w)}")

for e in dictionary:
    print(e)

print(len(allowed_entities))
print(len(dictionary))

