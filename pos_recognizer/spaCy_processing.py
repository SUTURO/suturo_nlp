#!/usr/bin/python3
import spacy
from openpyxl import Workbook

def main():
    file = open("/home/chris/Desktop/SUTURO/testcode/single_test.txt", "r")
    #file = open("/home/chris/Desktop/SUTURO/testcode/result_len.txt", "r")

    wb = Workbook()
    ws = wb.create_sheet("spaCy", 0)
    ws.title = "spaCy"
    ws.append(["Words", "Part of Speech", "Tag", "Dependency", "Sentence"])

    for line in file:
        doc = nlp(line)
        for token in doc:
            content = [token.text, token.pos_, token.tag_, token.dep_, sentence(token.text, doc)]  # Hier mehr token Stuff einfügen wenn gewünscht
            ws.append(content)
        ws.append([])

    wb.save("/home/chris/Desktop/SUTURO/testcode/spaCy.xlsx")

# Erkennt nur zwei Teilsätze, wenns drei gibt wird der zweite überschrieben
def sentence(wort, doc):
    sent = []
    satz,satz1 = [], []

    for token in doc:
        sent.append((token.text, token.pos))
    
    for word, pos in sent:
        if pos == 89:
            satz1 = satz
            satz = [word] 
        else:
            satz.append(word)


    if wort in satz and wort in satz1:
        return "Unclear"
    elif wort in satz:
        return " ".join(satz)
    elif wort in satz1:
        return " ".join(satz1)
    else:
        return "Null"

        
            

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    main()