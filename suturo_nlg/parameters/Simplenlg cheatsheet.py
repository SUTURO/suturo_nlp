import simplenlg
from simplenlg.lexicon import *
from simplenlg.framewok import *
from simplenlg.realiser.english import *
from simplenlg.phrasespec import *
from simplenlg.features import *

lexicon = Lexicon().getDefaultLexicon()
nlgFactory = NLGFactory(lexicon)
realiser = Realiser(lexicon)

p = nlgFactory.createClause()
p.setSubject("Mary")
p.setVerb("chase")
p.setObject("the monkey")
p.getObject().setDeterminer("the")
p.getObject().setPlural(True)
p.getObject().setPlural(False)
p.setFeature(Feature.PERFECT, True)

p.setFeature('tense', Tense.PRESENT)
p.setFeature('negated', True)

#present progressive tense?



output = realiser.realiseSentence(p)
