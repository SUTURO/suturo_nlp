#!/usr/bin/python3
# Parses text
import sling
from collections import defaultdict
# Used to communicate with ros via textparserRosInterface
import zmq

import sys

from tinyrpc.server import RPCServer
from tinyrpc.dispatch import RPCDispatcher
from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
from tinyrpc.transports.zmq import ZmqServerTransport

ctx = zmq.Context()
dispatcher = RPCDispatcher()
transport = ZmqServerTransport.create(ctx, 'tcp://127.0.0.1:5001')

rpc_server = RPCServer(
    transport,
    JSONRPCProtocol(),
    dispatcher
)

# parser used to parse/ interpred the given text
parser = sling.Parser(sys.argv[1] + "/../flow/qa_system.flow")


@dispatcher.public
def semantic_parse(text):
    doc = parser.parse(text)
    mapping = defaultdict(list)   # Frame -> Mention(s) that evoke it
    for mention in doc.mentions:
        for frame in mention.evokes():
            mapping[frame].append(mention)
    pred_frame = None
    for k in mapping.keys():        
        if k.isa("predicate"):
            pred_frame = k
            break
    if not pred_frame:
        return {}
    roles = []
    for k, v in pred_frame:
        roles.append(str(k)[2:-1])
    m = mapping[pred_frame][0]
    retq = {"predicate": doc.phrase(m.begin, m.end)}
    for r in roles:
        if "isa" == r:
            continue
        m = mapping[pred_frame[r]][0]
        retq[r] = doc.phrase(m.begin, m.end)
    return retq

# start the server and make it run forever
rpc_server.serve_forever()
