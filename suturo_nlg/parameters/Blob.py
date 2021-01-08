import simplenlg
from simplenlg.lexicon import *
from simplenlg.framework import *
from simplenlg.realiser.english import *
from simplenlg.phrasespec import *
from simplenlg.features import *
import random

lexicon = Lexicon().getDefaultLexicon()
nlgFactory = NLGFactory(lexicon)
realiser = Realiser(lexicon)

def randomString(strings):
    return strings[random.randint(0, len(strings) - 1)]

def describeTask(task, tense, progressive=True):
    p = nlgFactory.createClause()
    p.setFeature("tense", tense)
    p.setSubject("I")
    p.setVerb(task["action"])
    if "patient" in task:
        p.setObject(task["patient"])
        p.getObject().setDeterminer("the")
        patientCount = 1
        if "patientNumber" in task:
            patientCount = task["patientNumber"]
        if 0 == patientCount:
            p.getObject().setDeterminer("any")
            p.setFeature(Feature.NEGATED, True)
        p.getObject().setPlural((1 < patientCount))
    p.setFeature(Feature.PROGRESSIVE, progressive)
    if "destination" in task:
        place = nlgFactory.createNounPhrase(task["destination"])
        place.setDeterminer("the")
        pp = nlgFactory.createPrepositionPhrase()
        pp.addComplement(place)
        pp.setPreposition("to")
        p.addComplement(pp)
    return realiser.realiseSentence(p)

def processingTask(task, decision):
    if decision == "waiting":
        return randomString(["Calibrating", "Still calibrating", "Please wait"])
    elif decision == "failed":
        return randomString(["Does not compute", "Please repeat", "Please speak loud and clear", "Did not understand", "Repeat", "Try again", "Please rephrase", "Information is not compatible", "Contradicting information detected", "Corrupted information detected"])

print(processingTask({"action": "put", "patient": "book", "patientNumber":1, "destination": "shelf"}, randomString(["waiting", "waiting", "failed", "failed", "failed"])))
print(processingTask({"action": "put", "patient": "book", "patientNumber":1, "destination": "shelf"}, randomString(["waiting", "waiting", "failed", "failed", "failed"])))
print(processingTask({"action": "put", "patient": "book", "patientNumber":1, "destination": "shelf"}, randomString(["waiting", "waiting", "failed", "failed", "failed"])))
print(processingTask({"action": "put", "patient": "book", "patientNumber":1, "destination": "shelf"}, randomString(["waiting", "waiting", "failed", "failed", "failed"])))
print(processingTask({"action": "put", "patient": "book", "patientNumber":1, "destination": "shelf"}, randomString(["waiting", "waiting", "failed", "failed", "failed"])))
print(processingTask({"action": "put", "patient": "book", "patientNumber":1, "destination": "shelf"}, randomString(["waiting", "waiting", "failed", "failed", "failed"])))

print(processingTask({"action": "put", "patient": "book", "patientNumber":1, "destination": "shelf"}, randomString(["waiting", "waiting", "failed", "failed", "failed"])))
print(processingTask({"action": "put", "patient": "book", "patientNumber":1, "destination": "shelf"}, randomString(["waiting", "waiting", "failed", "failed", "failed"])))
print(processingTask({"action": "put", "patient": "book", "patientNumber":1, "destination": "shelf"}, randomString(["waiting", "waiting", "failed", "failed", "failed"])))
print(processingTask({"action": "put", "patient": "book", "patientNumber":1, "destination": "shelf"}, randomString(["waiting", "waiting", "failed", "failed", "failed"])))
print(processingTask({"action": "put", "patient": "book", "patientNumber":1, "destination": "shelf"}, randomString(["waiting", "waiting", "failed", "failed", "failed"])))
print(processingTask({"action": "put", "patient": "book", "patientNumber":1, "destination": "shelf"}, randomString(["waiting", "waiting", "failed", "failed", "failed"])))
