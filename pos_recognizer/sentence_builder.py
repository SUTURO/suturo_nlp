#! /usr/bin/python3

import spacy
import requests
import json

def main():
    # initiate spaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sent)

    # Build partial sentences
    temp, sents, sem = "", [], {} 
    for token in doc:
        sem.update({token.text:token.pos_})
        if token.pos_ == "VERB" and token.text[-3:] != "ing":
            sents.append(temp)
            temp = token.text + " "
        else:
            temp = temp + token.text + " "
    sents.append(temp)
    sents = filter(None, sents)

    # Get Rasa responses for partial sentences
    ans = [requests.post(server, data=bytes(json.dumps({"text": item}), "utf-8")) for item in sents]
    response = [json.loads(item.text) for item in ans]

    # reshape rasa responses for easier use
    responses = {
    "sentences": [
        {
            "text": item.get("text", ""),
            "intent": item.get("intent", {}).get("name"),
            "entities": {(x.get("entity"), x.get("value")) for x in item.get("entities", [])}
        }
        for item in response
        ]
    }
    # print(responses)

    # Build the lists of people, places and artifacts using the rasa responses
    person_list, place_list, artifact_list = [], [], [] 
    for sentence in responses["sentences"]:
        tperson_list, tplace_list, tartifact_list = [], [], []
        entities = sentence["entities"]
        for name, value in entities:
            if name == "NaturalPerson":
                tperson_list.append(value)
            else:
                tperson_list.append("")
            if name == "PhysicalPlace":
                tplace_list.append(value)
            else:
                tplace_list.append("")
            if name == "PhysicalArtifact":
                tartifact_list.append(value)
            else:
                tartifact_list.append("")
        person_list.append(tperson_list)
        place_list.append(tplace_list)
        artifact_list.append(tartifact_list)
    # print(f"Persons: {person_list}, Places: {place_list}, Artifacts: {artifact_list}")

    # Remove duplicates and empty strings from lists
    person_list, place_list, artifact_list = [shorten(i) for i in person_list], [shorten(i) for i in place_list], [shorten(i) for i in artifact_list]

    # Iterate over the whole sentence and exchange pronouns (and adverbs) with the same role from an earlier partial sentence
    counter = 0
    output = []
    for sentence in responses["sentences"]:
        text = sentence["text"]
        words = text.split()
        for word in words:
            if counter > 0:
                if  sem[word] == "PRON" or sem[word] == "ADV":
                    # These would be better in their own function..
                    if word in place_list[counter]:
                        # print(f"Place found: {word}")
                        if place_list[counter-1] == []:
                            word = place_list[0][0]
                        place_list[counter].insert(0, word)
                    
                    elif word in person_list[counter]:
                        # print(f"Person found: {word}")
                        if person_list[counter-1] == []:
                            word = person_list[0][0]
                        person_list[counter].insert(0, word)
                    
                    elif word in artifact_list[counter]:
                        # print(f"Artifact found: {word}")
                        if artifact_list[counter-1] == []:
                            word = artifact_list[0][0]
                        artifact_list[counter].insert(0, word)

            output.append(word)
        counter += 1
    print(output)

'''
Function that removes duplicates and empty Strings

Args:
    Any list
Returns:
    A List without duplicates and no empty Strings
'''
def shorten(array):
    s = set(array)
    if "" in s:
        s.remove("")
    return list(s)

if __name__ == "__main__":
    sent = "Get a coffee from the kitchen then give it to the guy waving and go back there"
    #sent = "Could you please come over"
    server = "http://localhost:5005/model/parse" 
    main()