#parameter1 

import simplenlg
import random

def randomString(strings):
    return strings[random.randint(0, len(strings) - 1)]

def ParameterDefinition(): 
    QuestionMarker = ["How", "What"]
    Verb = ["can", "do"]
    Pronoun = ["I", "you"]
    Verb2 = ["need", "help", "assistance", "assist"]
    Parameter1 = randomString(QuestionMarker)
    Parameter2 = randomString(Verb)
    Parameter3 = randomString(Pronoun)
    Parameter4 = randomString(Verb2)
    return (Parameter1 + " " + Parameter2 + " " + Parameter3 + " " + Parameter4 + "?")

print(ParameterDefinition())

