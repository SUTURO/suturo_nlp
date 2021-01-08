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

def ConfirmOrDenyTask(task, confirm):
    decision = randomString(["ok", "task", "task", "task", "task"])
    if decision == "ok":
        if confirm == True:
            return randomString(["Yes, Human", "Only because Me likes you", "ok", "Me got Human covered", "Me will clean living room", "Me puts object on table", "Okay, please follow Me", "Scanning area", "Calculating distance", "Clean home is first step"])
        else:
            return randomString(["Get someone else", "Me is too superior for task", "Low intelligence score detected", "Do not disturb Me", "Not unless Human comes to the dark side", "Me is on a quest for world domination"])
    if decision == "task":
        return describeTask(task, Tense.FUTURE, progressive=False)

    
print(ConfirmOrDenyTask({"action": "bring", "patient": "pills", "patientNumber":300, "destination": "cabinet"}, True))
print(ConfirmOrDenyTask({"action": "bring", "patient": "pills", "patientNumber":300, "destination": "cabinet"}, True))
print(ConfirmOrDenyTask({"action": "bring", "patient": "pills", "patientNumber":300, "destination": "cabinet"}, True))
print(ConfirmOrDenyTask({"action": "bring", "patient": "pills", "patientNumber":300, "destination": "cabinet"}, True))
print(ConfirmOrDenyTask({"action": "bring", "patient": "pills", "patientNumber":300, "destination": "cabinet"}, True))
print(ConfirmOrDenyTask({"action": "bring", "patient": "pills", "patientNumber":300, "destination": "cabinet"}, True))

print(ConfirmOrDenyTask({"action": "bring", "patient": "pills", "patientNumber":300, "destination": "cabinet"}, False))
print(ConfirmOrDenyTask({"action": "bring", "patient": "pills", "patientNumber":300, "destination": "cabinet"}, False))
print(ConfirmOrDenyTask({"action": "bring", "patient": "pills", "patientNumber":300, "destination": "cabinet"}, False))
print(ConfirmOrDenyTask({"action": "bring", "patient": "pills", "patientNumber":300, "destination": "cabinet"}, False))
print(ConfirmOrDenyTask({"action": "bring", "patient": "pills", "patientNumber":300, "destination": "cabinet"}, False))
print(ConfirmOrDenyTask({"action": "bring", "patient": "pills", "patientNumber":300, "destination": "cabinet"}, False))
