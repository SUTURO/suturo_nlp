import simplenlg
from simplenlg.lexicon import *
from simplenlg.framework import *
from simplenlg.realiser.english import *
from simplenlg.phrasespec import *
from simplenlg.features import *
import random
import rospy
from std_msgs.msg import String
from suturo_nlg.mgs import MeaningRepresentation 

lexicon = Lexicon().getDefaultLexicon()
nlgFactory = NLGFactory(lexicon)
realiser = Realiser(lexicon)
pub = rospy.Publisher('debug_textgen', String, queue_size=1)
rospy.init_node('talker', anonymous=True)

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

def TaskRoutine (task, decision):
    #decision = randomString (["ok", "task", "task", "task", "waiting", "waiting", "failed", "failed", "finished", "finished", "finished", "finished", "complications", "complications"])
    if decision == "reject":
        return randomString(["Get someone else. ", "Me is too superior for task. ", "Low intelligence score detected", "Do not disturb Me", "Not unless Human comes to the dark side", "Me is on a quest for world domination"])
    if decision == "task":
        prefix = randomString(["Yes, Human. ", "Only because Me likes you. ", "Ok. ", "Me got Human covered. "])
        #"Me will clean living room. ", "Me puts object on table. ", "Okay, please follow Me. ", "Scanning area. ", "Calculating distance. ", "Clean home is first step. "
        return prefix + describeTask(task, Tense.FUTURE, progressive=False)
    if decision == "waiting":
        return randomString(["Calibrating", "Still calibrating", "Please wait"])
    if decision == "failed":
        #"Does not compute", "Information is not compatible", "Contradicting information detected", "Corrupted information detected"
        return randomString(["Please repeat", "Please speak loud and clear", "Did not understand", "Repeat", "Try again", "Please rephrase"])
    if decision == "finished":
        return randomString (["Mischief managed", "Me is done", "Task completed"])
    if decision == "complications":
        return randomString (["Humans are too tall, Me cannot see", "There is an unknown object on the floor", "Object too tiny", "Me cannot pick up object", "There are too many Humans named Anna", "Me cannot reach this high", "pencil is too short for Me to pick up", "Me cannot go further", "Wall is blocking way"])
    return "huh?"

def callback(data):
    return

rospy.Subscriber("nlg_requests", MeaningRepresentation, callback)

for x in ["reject", "task", "waiting", "failed", "finished", "complications"]:
    pub.publish(TaskRoutine({"action": "bring", "patient": "pills", "patientNumber":300, "destination": "cabinet"}, x))

# spin() simply keeps python from exiting until this node is stopped
rospy.spin()
