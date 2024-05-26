#! /usr/bin/python3

import spacy
import requests
import json

def main():
    # initiate spaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sent)

    # Build partial sentences
    # Initiate variables: Temporary partial sentence, List of Sentences, Dictionary for every word with "part of speech" as key, 
    '''
    temp: Temporary partial sentence
    sents: List of partial sentences
    sem: Dictionary for every word with "part of speech" as key, used for iterating over words later
    global splits: List of verbs in the Sentence, used for splitting
    '''
    global sem, splits
    temp, sents, sem, splits = "", [], {}, [] 
    for token in doc:
        sem.update({token.text:token.pos_})
        if token.pos_ == "VERB" and token.text[-3:] != "ing":
            sents.append(temp)
            temp = token.text + " "
            splits.append(token.text)
        else:
            temp = temp + token.text + " "
    sents.append(temp)
    sents = filter(None, sents)

    # Get Rasa responses for partial sentences
    ans = [requests.post(server, data=bytes(json.dumps({"text": item}), "utf-8")) for item in sents]
    response = [json.loads(item.text) for item in ans]
    print(ans)
    print(response)

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
    print(len(responses.get("sentences")))
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

    # Manually remove "then" from place_list...
    i = 0
    for l in place_list:
        place_list[i] = filter(lambda word: word != "then", l)
        i+=1


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
                    # Repeating this for every list
                    if word in place_list[counter]:
                        if place_list[counter-1] == []:
                            word = place_list[0][0] if place_list[0] else word # don't do anything if list is empty
                        else:
                            word = place_list[counter-1][0]
                            place_list[counter].insert(0, word)
                    
                    elif word in person_list[counter]:
                        if person_list[counter-1] == []:
                            word = person_list[0][0] if person_list[0] else word
                        else:
                            word = person_list[counter-1][0]
                            person_list[counter].insert(0, word)
                    
                    elif word in artifact_list[counter]:
                        if artifact_list[counter-1] == []:
                            word = artifact_list[0][0] if artifact_list[0] else word
                        else:
                            word = artifact_list[counter-1][0]
                            artifact_list[counter].insert(0, word)

            output.append(word)
        counter += 1

    output = split_into_queue(splits, output)
    print(list(output)) # Output for gpsr, list method only for print statement

def shorten(array):
    '''
    Function that removes duplicates and empty Strings

    Args:
        array: Any list
    Returns:
        A List without duplicates and no empty Strings
    '''
    s = set(array)
    if "" in s:
        s.remove("")
    return list(s)


def split_into_queue(verbs, sent):
    '''
    Function that splits a sentence into parts at specific words.

    Args:
        verbs: A list of Strings
        sent: A String
    Returns:

    '''
    temp = ""
    out = []
    for word in sent:
        if word in verbs:
            out.append(temp)
            temp = word
        else:
            temp += f" {word}"
    out.append(temp)
    out = filter(None, out)
    return out

if __name__ == "__main__":
    # sent = "Get a coffee from the kitchen then give it to the guy waving and go back there"
    # sent = "Move the milk from the kitchen to the dining room then get the fork from there and bring it back"
    sent = "Locate a dice in the living room then fetch it and give it to Charlie in the living room"
    # sent =  "Bring me the milk"
    server = "http://localhost:5005/model/parse" 
    main()