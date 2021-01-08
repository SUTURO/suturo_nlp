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


@dispatcher.public
def generate_text(kvp_dict):
    if "error" in kvp_dict:
        return kvp_dict["error"]

    if not("question" in kvp_dict):
        return "huh?"

    query = kvp_dict["question"]

    if "is there" == query or "Is there" == query:
        if "yes" == kvp_dict["answer"]:
            return "yes"
        else:
            return "no"
    elif query == "supposed to go":
        item = kvp_dict["item"]
        location = kvp_dict["location"]
        return "the %s is supposed to go on the %s" % (item, location)
    elif query == "what is on":
        item = kvp_dict["item"]
        location = kvp_dict["location"]
        return "there is a %s on the %s" % (item, location)

    return "huh?"


rpc_server.serve_forever()


