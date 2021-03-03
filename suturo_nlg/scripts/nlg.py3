#!/usr/bin/env python3

from simplenlg.framework import *
from simplenlg.realiser.english import *
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
    kvp_dict = dict()
    for kv_tuple in list_of_tuples:
        kvp_dict[kv_tuple[0]] = kv_tuple[1]

    # count the amount of main keys
    # main keys are action,can't move, starting, stoping
    counter = 0
    for k in kvp_dict.keys():
        if k in ["action", "cantmove", "starting"]:
            counter = counter + 1

    if counter > 1:
        ret = error_msgs("More than one main key was given.")
    elif "action" in kvp_dict:
        ret = action_sentence(kvp_dict)
    elif "cantmove" in kvp_dict:
        ret = cantmove_sentence(kvp_dict)
    elif "starting" in kvp_dict:
        ret = starting_sentence(kvp_dict)
    elif "stopping" in kvp_dict:
        ret = stopping_sentence(kvp_dict)
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
        return "STARTING but didn't specify what procedure, please watch me!"

    phrase_spec.setComplement(", please watch me")
    return realiser.realise(phrase_spec)


# stopping sentence
#
def stopping_sentence(kvp_dict):
    return error_msgs("Stopping Sentences are not supported yet.")


# when the sentence is an action
#
def action_sentence(kvp_dict):
    # "place", "pickup", "move", "percieve"
    if kvp_dict["action"] == "place":
        return place_sentence(kvp_dict)
    elif kvp_dict["action"] == "pick up":
        return pickup_sentence(kvp_dict)
    elif kvp_dict["action"] == "percieve":
        return percieve_sentence(kvp_dict)
    elif kvp_dict["action"] == "move":
        return move_sentence(kvp_dict)
    return error_msgs("The Action " + kvp_dict["action"] + "is not known")


def place_sentence(kvp_dict):
    parent_clause = nlgFactory.createClause()
    parent_clause.setSubject("I")
    parent_clause.setTense(Tense.FUTURE)
    parent_clause.setVerb("place")
    parent_clause.setObject(kvp_dict["object_id"])
    if "goal_surface_id" in kvp_dict:
        prepositon = nlgFactory.createPrepositionPhrase()
        prepositon.setPreposition("on")
        prepositon.setObject(kvp_dict["goal_surface_id"])
        if "goal_room_id" in kvp_dict:
            prepositon.addComplement("in " + kvp_dict["goal_room_id"])
        parent_clause.addComplement(prepositon)
    if "reason" in kvp_dict:
        reason = nlgFactory.createClause()
        reason.setFeature("complementiser", ", because")
        reason.setSubject("it")
        if kvp_dict["reason"] == "similar color" or kvp_dict["reason"] == "similar size":
            reason.setVerb("be of " + kvp_dict["reason"] + " to")
        elif kvp_dict["reason"] == "similar type":
            reason.setVerb("be similar to")
        reason.setObject(kvp_dict["object_id_2"])
        parent_clause.addComplement(reason)

    return realiser.realise(parent_clause)


def pickup_sentence(kvp_dict):
    parent_clause = nlgFactory.createClause()
    parent_clause.setSubject("I")
    parent_clause.setTense(Tense.FUTURE)
    parent_clause.setVerb("pick up")
    parent_clause.setObject(kvp_dict["object_id"])
    if "goal_surface_id" in kvp_dict:
        parent_clause.addComplement("from the " + kvp_dict["goal_surface_id"])
    if "goal_room_id" in kvp_dict:
        parent_clause.addComplement("in the " + kvp_dict["goal_room_id"])
    return realiser.realise(parent_clause)


def percieve_sentence(kvp_dict):
    parent_clause = nlgFactory.createClause()
    parent_clause.setSubject("I")
    parent_clause.setTense(Tense.FUTURE)
    parent_clause.setVerb("look at")
    object_surface_room_order(kvp_dict, parent_clause)
    return realiser.realise(parent_clause)

def move_sentence(kvp_dict):
    parent_clause = nlgFactory.createClause()
    parent_clause.setSubject("I")
    parent_clause.setTense(Tense.FUTURE)
    parent_clause.setVerb("move")
    if "object_id" in kvp_dict:
        parent_clause.setObject(kvp_dict["object_id"])

    if "start_room_id" in kvp_dict and "start_surface_id" in kvp_dict:
        frompp = nlgFactory.createPrepositionPhrase()
        frompp.setPreposition("from")
        frompp.setObject(kvp_dict["start_surface_id"])
        frompp2 = nlgFactory.createPrepositionPhrase()
        frompp2.setPreposition("in")
        frompp2.setObject(kvp_dict["start_room_id"])
        parent_clause.addComplement(frompp)
        parent_clause.addComplement(frompp2)
    elif "start_room_id" in kvp_dict:
        frompp = nlgFactory.createPrepositionPhrase()
        frompp.setPreposition("from")
        frompp.setObject(kvp_dict["start_room_id"])
        parent_clause.addComplement(frompp)
    elif "start_surface_id" in kvp_dict:
        frompp = nlgFactory.createPrepositionPhrase()
        frompp.setPreposition("from")
        frompp.setObject(kvp_dict["start_surface_id"])
        parent_clause.addComplenent(frompp)
    parent_clause.addComplement("to")
    object_surface_room_order(kvp_dict, parent_clause)
    return realiser.realise(parent_clause)


# when the sentence is an cantmove
def cantmove_sentence(kvp_dict):
    # "reach", "too small", "too big", "too heavy"
    phrase_spec = nlgFactory.createClause()
    phrase_spec.setSubject("I")
    phrase_spec.setNegated(True)
    phrase_spec.setObject(kvp_dict["object_id"])
    phrase_spec.setTense(Tense.PRESENT)

    if kvp_dict["cantmove"] == "reach":
        phrase_spec.setVerb("can reach")
        if "start_surface_id" in kvp_dict:
            phrase_spec.setComplement("on " + kvp_dict["start_surface_id"])

    elif kvp_dict["cantmove"] == "too small" or kvp_dict["cantmove"] == "too big" or kvp_dict["cantmove"] == "too heavy":
        phrase_spec.setVerb("can grab")
        s2 = nlgFactory.createClause()
        s2.setSubject("it")
        s2.setVerb("be")
        s2.addModifier(kvp_dict["cantmove"])
        s2.setFeature("complementiser", ", because")
        phrase_spec.addComplement(s2)
    return realiser.realise(phrase_spec)


# Used by action_sentence
def object_surface_room_order(kvp_dict, phrase_spec):
    if "goal_surface_id" in kvp_dict and "object_id_2" in kvp_dict and "goal_room_id" in kvp_dict:
        phrase_spec.addComplement(kvp_dict["object_id_2"])
        phrase_spec.addComplement("on " + kvp_dict["goal_surface_id"])
        add_in_the_room(kvp_dict, phrase_spec)
    elif "goal_surface_id" in kvp_dict and "object_id_2" in kvp_dict:
        phrase_spec.addComplement(kvp_dict["object_id_2"])
        phrase_spec.addComplement("on " + kvp_dict["goal_surface_id"])
    elif "goal_surface_id" in kvp_dict and "goal_room_id" in kvp_dict:
        phrase_spec.addComplement(kvp_dict["goal_surface_id"])
        phrase_spec.addComplement("in " + kvp_dict["goal_room_id"])
    elif "object_id_2" in kvp_dict and "goal_room_id" in kvp_dict:
        phrase_spec.addComplement(kvp_dict["object_id_2"])
        phrase_spec.addComplement("in " + kvp_dict["goal_room_id"])
    elif "goal_surface_id" in kvp_dict:
        phrase_spec.addComplement(kvp_dict["goal_surface_id"])
    elif "object_id_2" in kvp_dict:
        phrase_spec.addComplement(kvp_dict["object_id_2"])
    elif "goal_room_id" in kvp_dict:
        phrase_spec.addComplement(kvp_dict["goal_room_id"])


def add_in_the_room(kvp_dict, phrase_spec):
    if "goal_room_id" in kvp_dict:
        phrase_spec.addComplement("in the " + kvp_dict["goal_room_id"])


rpc_server.serve_forever()
