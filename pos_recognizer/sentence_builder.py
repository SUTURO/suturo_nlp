#! /usr/bin/python3

import spacy
import requests
import json

def deprecated():
    pass
    '''
    class Tree:
        def __init__(self, data):
            self.children = []
            self.data = data

    def print_tree(node, depth=0):
        indent = "  " * depth
        print(f"{indent}{node.data}")
        for child in node.children:
            print_tree(child, depth + 1)
    
    def add_subsentences(token, parent_node):
        for child in token.children:
            if child.pos_ != "VERB":
                node = Tree(child.text)
                parent_node.children.append(node)
                add_subsentences(child, node)
            else:
                break

    root = Tree("Tree")

    for token in doc:
        if token.pos_ == "VERB":
            node = Tree(token.text)
            add_subsentences(token, node)
            root.children.append(node)
            
    print_tree(root)
    '''

def main():
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(sent)

    temp, sents, sem = "", [], {}
    for token in doc:
        sem.update({token.text:token.pos_})
        if token.pos_ == "VERB":
            sents.append(temp)
            temp = token.text + " "
        else:
            temp = temp + token.text + " "
    sents.append(temp)

    sents = filter(None, sents)

    ans = [requests.post(server, data=bytes(json.dumps({"text": item}), "utf-8")) for item in sents]

    response = [json.loads(item.text) for item in ans]

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

    print(responses)
    person_list, place_list, artifact_list = [], [], []
    #Funktion um variablen zu setzen
    for sentence in responses["sentences"]:
        entities = sentence["entities"]
        for name, value in entities:
            if name == "NaturalPerson":
                person_list.append(value)
            elif name == "PhysicalPlace":
                place_list = value
            elif name == "PhysicalArtifact":
                artifact_list.append(value)

    print(f"Person: {person_list}, Place: {place_list}, Artifacts: {artifact_list}")
    for sentence in responses["sentences"]:
        text = sentence["text"]
        words = text.split()
        for word in words:
            if  sem[word] == "PRON":    # also check role
                print(word)
                word = person_list[0]   # muss das direkte wort überschreiben nicht die variable
                


    # Jetzt über alle wörter gehen und vergleichen, ob sem(wort)==PRON, wenn ja ersetzen durch physicalartifact oder naturalperson, wenn nein weiter
    #for word in responses["sentences": [{"text"}]]:
    #    print(word)
    #    if sem[word] == "PRON":
    #        word = person_list[0]


if __name__ == "__main__":
    sent = "Bring me the fork and the Spoon from the kitchen then place them on the table and get the food from the garage."
    #sent = "Could you please come over"
    server = "http://localhost:5005/model/parse" 
    main()