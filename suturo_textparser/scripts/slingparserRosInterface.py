#!/usr/bin/env python
import rospy
import rospkg
from suturo_nlg.msg import KeyValuePair, MeaningRepresentation
from std_msgs.msg import String

import subprocess
import os

import zmq
from tinyrpc import RPCClient
from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
from tinyrpc.transports.zmq import ZmqClientTransport


ctx = zmq.Context()

rpc_client = RPCClient(
    JSONRPCProtocol(),
    ZmqClientTransport.create(ctx, 'tcp://127.0.0.1:5001')
)

str_server = rpc_client.get_proxy()


def callback(data):
    input_str = data.data

    input_array = input_str.split('\\')

    for in_str in input_array:
        parsedDict = str_server.semantic_parse(in_str)
        mr = MeaningRepresentation()
        for key in parsedDict.keys():
            kvp = KeyValuePair()
            kvp.key = key
            kvp.value = parsedDict[key]
            mr.role_values.append(kvp)
        pub.publish(mr)


if __name__ == '__main__':
    rospack = rospkg.RosPack()
    wd = rospack.get_path('suturo_textparser')
    try:
        rospy.loginfo('trying')
        # Start the publisher node
        rospy.init_node('tpri', anonymous=True)
        pub = rospy.Publisher('sp_output', MeaningRepresentation, queue_size=10)
        # Start the subscriber node
        rospy.Subscriber('sp_input', String, callback)
        subprocess.call('python3 scripts/slingparser.py3 ' + wd + '/scripts', cwd=wd, shell=True)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
