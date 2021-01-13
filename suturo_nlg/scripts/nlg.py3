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

import json

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
def generate_text(list_of_tuples):
    dictionary = dict()
    
    for kv_tuple in list_of_tuples:
        dictionary[kv_tuple[0]] = kv_tuple[1]
    
    print (dictionary)
    if ("action" in dictionary) and ("cantmove" in dictionary):
        #TODO THROW ERROR
        print("error1")
    elif ("action" in dictionary):
        # "place", "pickup", "move", "look at"
        return action_sentence(dictionary)            
    elif ("cantmove" in dictionary):
        print("error2")
    else:
        #TODO THROW ERROR
        print("error3")



# when the sentence is an action
def action_sentence(dictionary):
    # "place", "pickup", "move", "look at"   
    sphraseSpec = nlgFactory.createClause()
    sphraseSpec.setSubject("I")
    sphraseSpec.setTense(Tense.FUTURE)

    if dictionary["action"] == "place" :
        sphraseSpec.setVerb("place")
        sphraseSpec.setObject(dictionary["object_id"])
        if "goal_surface_id" in dictionary:
            sphraseSpec.addComplement("on the " + dictionary["goal_surface_id"])
        add_in_the_room(dictionary, sphraseSpec)

    elif dictionary["action"] == "pick up" :
        sphraseSpec.setVerb("pick up")         
        sphraseSpec.setObject(dictionary["object_id"])
        if "goal_surface_id" in dictionary:
            sphraseSpec.addComplement("from the " + dictionary["goal_surface_id"])
        add_in_the_room(dictionary, sphraseSpec)

    elif dictionary["action"] == "look at" :
        sphraseSpec.setVerb("look at")     
        object_surface_room_order(dictionary, sphraseSpec)

    elif dictionary["action"] == "move" :
        sphraseSpec.setVerb("move to")             
        object_surface_room_order(dictionary, sphraseSpec)
    
    print(realiser.realise(sphraseSpec))
    return str(realiser.realise(sphraseSpec))



# Used by action_sentence
def object_surface_room_order(dictionary, sphraseSpec):
    if "goal_surface_id" in dictionary and "object_id" in dictionary and "goal_room_id" in dictionary:
        sphraseSpec.addComplement("the " + dictionary["object_id"])
        sphraseSpec.addComplement("on the " + dictionary["goal_surface_id"])
        add_in_the_room(dictionary, sphraseSpec)
    elif "goal_surface_id" in dictionary and "object_id" in dictionary:
        sphraseSpec.addComplement("the " + dictionary["object_id"])
        sphraseSpec.addComplement("on the " + dictionary["goal_surface_id"])
    elif "goal_surface_id" in dictionary and "goal_room_id" in dictionary:
        sphraseSpec.addComplement("the " + dictionary["goal_surface_id"])
        sphraseSpec.addComplement("on the " + dictionary["goal_room_id"])
    elif "object_id" in dictionary and "goal_room_id" in dictionary:
        sphraseSpec.addComplement("the " + dictionary["object_id"])
        sphraseSpec.addComplement("on the " + dictionary["goal_room_id"])
    elif "goal_surface_id" in dictionary:
        sphraseSpec.addComplement("the " + dictionary["goal_surface_id"])
    elif "object_id" in dictionary:
        sphraseSpec.addComplement("the " + dictionary["object_id"])
    elif "goal_room_id" in dictionary:
        sphraseSpec.addComplement("the " + dictionary["goal_room_id"])


def add_in_the_room(dictionary, sphraseSpec):
    if "goal_room_id" in dictionary:
        sphraseSpec.addComplement("in the " + dictionary["goal_room_id"])

rpc_server.serve_forever()

