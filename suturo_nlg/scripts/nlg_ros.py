#!/usr/bin/env python
# generall ROS stuff and MSGS

import rospy
from std_msgs.msg import String
from nlg_msgs.msg import MeaningRepresentation, KeyValuePair
import tmc_msgs.msg
from tmc_msgs.msg import Voice

# connection to nlg.py3 script
import zmq
from tinyrpc import RPCClient, RPCProxy
from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
from tinyrpc.transports.zmq import ZmqClientTransport

rpc_client = RPCClient(
    JSONRPCProtocol(),
    ZmqClientTransport.create(zmq.Context(), 'tcp://127.0.0.1:5002')
)
gen_server = None


def callback(data):
    # create String
    list_send = []
    for kvp in data.role_values:
        # TODO get Object/Surface spoken name
        list_send.append((kvp.key, kvp.value))

    value_from_s_nlg = gen_server.generate_text(list_send)
    if value_from_s_nlg.startswith("!ERROR!~"):
        error_msg = value_from_s_nlg.split("~")[1]
        rospy.logerror(error_msg)
        pub_sentence = "An error accured in the natural language generation pipeline. The error given was: " + error_msg
    else:
        pub_sentence = value_from_s_nlg

        textrenderer_pub.publish(pub_sentence)
        talk_request_pub.publish(Voice(sentence=pub_sentence, language=tmc_msgs.msg.Voice.kEnglish))
    # taken from /hsrb_battery_notifier/battery_notifier.py I don't know what the k stands for either


if __name__ == '__main__':
    talk_request_pub = rospy.Publisher('/talk_request', Voice, queue_size=1)
    textrenderer_pub = rospy.Publisher('/textrenderer', String, queue_size=1) # for gazebo
    rospy.Subscriber("nlg_requests", MeaningRepresentation, callback)
    rospy.init_node('natural_language_generator', anonymous=True)
    rospy.loginfo("NLG_ROS: Connecting to nlg.py3")
    gen_server = rpc_client.get_proxy()
    rospy.loginfo(gen_server.test_connection())
    # run forever and wait for incoming msgs
    rospy.spin()

