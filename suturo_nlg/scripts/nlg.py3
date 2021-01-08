#!/usr/bin/env python3
import simplenlg
from simplenlg.lexicon import *
from simplenlg.framework import *
from simplenlg.realiser.english import *
from simplenlg.phrasespec import *
from simplenlg.features import *
import random
import time
import zmq
from tinyrpc.server import RPCServer
from tinyrpc.dispatch import RPCDispatcher
from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
from tinyrpc.transports.zmq import ZmqServerTransport

ctx = zmq.Context()
dispatcher = RPCDispatcher()
transport = ZmqServerTransport.create(ctx, 'tcp://127.0.0.1:5002')

rpc_server = RPCServer(
    transport,
    JSONRPCProtocol(),
    dispatcher
)

lexicon = Lexicon().getDefaultLexicon()
nlgFactory = NLGFactory(lexicon)
realiser = Realiser(lexicon)

print("Oi! Started.")

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
    if decision == "reject": #The robot denies starting the task.
        return randomString(["Get someone else. ", "Me is too superior for task. ", "Low intelligence score detected", "Do not disturb Me", "Not unless Human comes to the dark side", "Me is on a quest for world domination"])
    if decision == "task": #The Robot agrees to completing the given task.
        prefix = randomString(["Yes, Human. ", "Only because Me likes you. ", "Ok. ", "Me got Human covered. "])
        #"Me will clean living room. ", "Me puts object on table. ", "Okay, please follow Me. ", "Scanning area. ", "Calculating distance. ", "Clean home is first step. "
        return prefix + describeTask(task, Tense.FUTURE, progressive=False)
    if decision == "waiting": #The Robot is waiting ot understand the information it was given
        return randomString(["Calibrating", "Still calibrating", "Please wait"])
    if decision == "failed": #The waiting process failed and the robot will ask you to ask again or rephrase the task or question
        #"Does not compute", "Information is not compatible", "Contradicting information detected", "Corrupted information detected"
        return randomString(["Please repeat", "Please speak loud and clear", "Did not understand", "Repeat", "Try again", "Please rephrase"])
    if decision == "finished": #The Robot finished the given task
        return randomString (["Mischief managed", "Me is done", "Task completed"])
    if decision == "complications": #The robot encountered complications during the performance of the given task.
        return randomString (["Humans are too tall, Me cannot see", "There is an unknown object on the floor", "Object too tiny", "Me cannot pick up object", "There are too many Humans named Anna", "Me cannot reach this high", "pencil is too short for Me to pick up", "Me cannot go further", "Wall is blocking way"])
    return "huh?"

@dispatcher.public
def generate_text(data):
    print("BOO")
    dataPie = data
    message = "huh?"
    if "failure" in dataPie:
        1 #TODO: describe failure
    elif "action" in dataPie:
        if "decision" in dataPie:
            decision = dataPie.pop("decision")
        else:
            decision = "task"
        message=(TaskRoutine(dataPie, decision))
    return message

rpc_server.serve_forever()

