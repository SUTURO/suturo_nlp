#!/usr/bin/env python
import rospy
import rospkg
from std_msgs.msg import String
from suturo_nlg.msg import MeaningRepresentation, KeyValuePair

import subprocess
import os

import time

import zmq
from tinyrpc import RPCClient
from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
from tinyrpc.transports.zmq import ZmqClientTransport

from tmc_msgs.msg import Voice, TalkRequestGoal, TalkRequestActionGoal, TalkRequestAction
import actionlib

ctx = zmq.Context()

rpc_client = RPCClient(
    JSONRPCProtocol(),
    ZmqClientTransport.create(ctx, 'tcp://127.0.0.1:5002')
)

str_server = rpc_client.get_proxy()


def callback(data):
    dataPie = {}
    for rv in data.role_values:
        isNumber = (rv.key.rfind("Number") != -1)
        if isNumber:
            dataPie[rv.key] = int(rv.value)
        else:
            dataPie[rv.key] = rv.value
    print(dataPie)
    message = str_server.generate_text(dataPie)
    pub.publish(message)
    if "True" == rospy.get_param('~use_action'):
        client = actionlib.SimpleActionClient('talk_request_action', TalkRequestAction)
        client.wait_for_server()
        goal = TalkRequestGoal(data=Voice(language=1, sentence=message))
        client.send_goal(goal)
        client.wait_for_result()


if __name__ == '__main__':
    rospy.init_node('talker', anonymous=True)
    pub = rospy.Publisher('debug_textgen', String, queue_size=1)
    rospy.Subscriber("nlg_requests", MeaningRepresentation, callback)
    rospack = rospkg.RosPack()
    wd = rospack.get_path('suturo_nlg') + "/scripts"
    subprocess.call('python3 nlg.py3', cwd=wd, shell=True)
    rospy.spin()
