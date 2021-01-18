#!/usr/bin/env python3

# from simplenlg.lexicon import *
from simplenlg.framework import *
from simplenlg.realiser.english import *
# from simplenlg.phrasespec import *
from simplenlg.features import *

import zmq
from tinyrpc.server import RPCServer
from tinyrpc.dispatch import RPCDispatcher
from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
from tinyrpc.transports.zmq import ZmqServerTransport


# setup the rpc connection
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


# This is called by the ros interface
@dispatcher.public
def test_connection():
    return "Connection Success"


# This is called by the ros interface
@dispatcher.public
def generate_text(list_of_tuples):
    # convert list of tuples to dict
    dictionary = dict()
    for kv_tuple in list_of_tuples:
        dictionary[kv_tuple[0]] = kv_tuple[1]

    # count the amount of main keys
    # main keys are action,can't move, starting, stoping
    counter = 0
    for k in dictionary.keys():
        if k in ["action", "cantmove", "starting"]:
            counter = counter + 1

    if counter > 1:
        ret = error_msgs("More than one main key was given.")
    elif "action" in dictionary:
        ret = action_sentence(dictionary)
    elif "cantmove" in dictionary:
        ret = cantmove_sentence(dictionary)
    elif "starting" in dictionary:
        ret = starting_sentence(dictionary)
    elif "stopping" in dictionary:
        ret = stopping_sentence(dictionary)
    else:
        ret = error_msgs("no main key was given.")
    return str(ret)


def error_msgs(string):
    return "!ERROR!~" + string  # can be easily cut at ~


# starting sentence
#
def starting_sentence(dictionary):
    # "clean up", "groceries", "safety test"
    phrase_spec = nlgFactory.createClause()
    phrase_spec.setSubject("I")
    phrase_spec.setTense(Tense.FUTURE)
    if dictionary["starting"] == "clean up":
        phrase_spec.setVerb("start to clean up")
        if "goal_room_id" in dictionary:
            phrase_spec.setObject(dictionary["goal_room_id"])
    elif dictionary["starting"] == "groceries":
        phrase_spec.setVerb("start to store the groceries")
    elif dictionary["starting"] == "safety test":
        phrase_spec.setVerb("start the safty test procedure")
    else:
        return "STARTING but didn't specify what procedure, please whatch me"

    phrase_spec.setComplement(", please whatch me")
    return realiser.realise(phrase_spec)


# stopping sentence
#
def stopping_sentence(dictionary):
    return error_msgs("Stopping Sentences are not supported yet.")


# when the sentence is an action
#
def action_sentence(dictionary):
    # "place", "pickup", "move", "percieve"   
    phrase_spec = nlgFactory.createClause()
    phrase_spec.setSubject("I")
    phrase_spec.setTense(Tense.FUTURE)

    # TODO Use Reason
    if dictionary["action"] == "place":
        phrase_spec.setVerb("place")
        phrase_spec.setObject(dictionary["object_id"])
        if "goal_surface_id" in dictionary:
            phrase_spec.addComplement("on the " + dictionary["goal_surface_id"])
            add_in_the_room(dictionary, phrase_spec)
            s2 = nlgFactory.createClause()
            s2.setFeature("complementiser", ", because")
            s2.setSubject("it")
            # "free space", "similar type", "similar color", "similar size"
            if dictionary["reason"] == "similar color" or dictionary["reason"] == "similar size":
                s2.setVerb("be of " + dictionary["reason"] + "to")
                s2.setObject(dictionary["object_id_2"])
            elif dictionary["reason"] == "similar type":
                s2.setVerb("be similar to")
                s2.setObject(dictionary["object_id_2"])
            phrase_spec.addComplement(s2)

    elif dictionary["action"] == "pick up":
        phrase_spec.setVerb("pick up")         
        phrase_spec.setObject(dictionary["object_id"])
        if "goal_surface_id" in dictionary:
            phrase_spec.addComplement("from the " + dictionary["goal_surface_id"])
        add_in_the_room(dictionary, phrase_spec)

    elif dictionary["action"] == "percieve":
        phrase_spec.setVerb("look at")     
        object_surface_room_order(dictionary, phrase_spec)

    elif dictionary["action"] == "move":
        phrase_spec.setVerb("move")
        if "object_id" in dictionary:
            phrase_spec.setObject(dictionary["object_id"])
        phrase_spec.addComplement("to")
        object_surface_room_order(dictionary, phrase_spec)

    return realiser.realise(phrase_spec)


# when the sentence is an cantmove
def cantmove_sentence(dictionary):
    # "reach", "too small", "too big", "too heavy"
    phrase_spec = nlgFactory.createClause()
    phrase_spec.setSubject("I")
    phrase_spec.setNegated(True)
    phrase_spec.setObject(dictionary["object_id"])
    phrase_spec.setTense(Tense.PRESENT)

    if dictionary["cantmove"] == "reach":
        phrase_spec.setVerb("can reach")

    elif dictionary["cantmove"] == "too small" or dictionary["cantmove"] == "too big" or dictionary["cantmove"] == "too heavy":
        phrase_spec.setVerb("can grab")
        s2 = nlgFactory.createClause()
        s2.setSubject("it")
        s2.setVerb("be")
        s2.addModifier(dictionary["cantmove"])
        s2.setFeature("complementiser", ", because")
        phrase_spec.addComplement(s2)
    return realiser.realise(phrase_spec)


# Used by action_sentence
def object_surface_room_order(dictionary, phrase_spec):
    if "goal_surface_id" in dictionary and "object_id_2" in dictionary and "goal_room_id" in dictionary:
        phrase_spec.addComplement(dictionary["object_id_2"])
        phrase_spec.addComplement("on " + dictionary["goal_surface_id"])
        add_in_the_room(dictionary, phrase_spec)
    elif "goal_surface_id" in dictionary and "object_id_2" in dictionary:
        phrase_spec.addComplement(dictionary["object_id_2"])
        phrase_spec.addComplement("on " + dictionary["goal_surface_id"])
    elif "goal_surface_id" in dictionary and "goal_room_id" in dictionary:
        phrase_spec.addComplement(dictionary["goal_surface_id"])
        phrase_spec.addComplement("on " + dictionary["goal_room_id"])
    elif "object_id_2" in dictionary and "goal_room_id" in dictionary:
        phrase_spec.addComplement(dictionary["object_id_2"])
        phrase_spec.addComplement("on " + dictionary["goal_room_id"])
    elif "goal_surface_id" in dictionary:
        phrase_spec.addComplement(dictionary["goal_surface_id"])
    elif "object_id_2" in dictionary:
        phrase_spec.addComplement(dictionary["object_id_2"])
    elif "goal_room_id" in dictionary:
        phrase_spec.addComplement(dictionary["goal_room_id"])


def add_in_the_room(dictionary, phrase_spec):
    if "goal_room_id" in dictionary:
        phrase_spec.addComplement("in the " + dictionary["goal_room_id"])


rpc_server.serve_forever()
