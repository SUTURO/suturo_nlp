import simplenlg
from simplenlg.lexicon import *
from simplenlg.framework import *
from simplenlg.realiser.english import *
from simplenlg.phrasespec import *
from simplenlg.features import *

lexicon = Lexicon().getDefaultLexicon()
nlgFactory = NLGFactory(lexicon)
realiser = Realiser(lexicon)

def makeProgressiveTowards(subject, action, patient, destination, patientCount=1):
    p = nlgFactory.createClause()
    p.setSubject(subject)
    p.setVerb(action)
    p.setObject(patient)
    p.getObject().setDeterminer("the")
    if 0 == patientCount:
        p.getObject().setDeterminer("any")
        p.setFeature(Feature.NEGATED, True)
    p.getObject().setPlural((1 < patientCount))
    p.setFeature(Feature.PROGRESSIVE, True)
    place = nlgFactory.createNounPhrase(destination)
    place.setDeterminer("the")
    pp = nlgFactory.createPrepositionPhrase()
    pp.addComplement(place)
    pp.setPreposition("to")
    p.addComplement(pp)
    return realiser.realiseSentence(p)

print(makeProgressiveTowards('I', 'carry', 'cup', 'table', patientCount=0))
print(makeProgressiveTowards('I', 'carry', 'cup', 'table'))
print(makeProgressiveTowards('I', 'carry', 'cup', 'table', patientCount=2))
